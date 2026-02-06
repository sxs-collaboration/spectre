// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/ModalSpacetimeInterpolator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/version.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/H5/Wrappers.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Legendre.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel//Printf/Printf.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

namespace {

// NOLINTNEXTLINE(clang-diagnostic-missing-noreturn)
void check_boost_version() {
  // volatile to avoid unused variable warning here and in constructors
  const volatile int boost_version = BOOST_VERSION;
  if (boost_version < 108100) {
    ERROR("ModalSpacetimeInterpolator requires Boost 1.81 or newer, but found "
          << BOOST_LIB_VERSION << ".");
  }
}

std::vector<std::string> resolve_filenames(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob) {
  std::vector<std::string> filenames =
      std::visit(Overloader{[](const std::vector<std::string>& volume_files) {
                              return volume_files;
                            },
                            [](const std::string& volume_files_glob) {
                              return file_system::glob(volume_files_glob);
                            }},
                 volume_files_or_glob);
  if (filenames.empty()) {
    ERROR("No volume files found. Specify at least one volume file.");
  }
  return filenames;
}

std::vector<std::pair<size_t, double>> load_observation_ids(
    const std::vector<std::string>& filenames, const std::string& subfile_name,
    const double start_time, const double end_time) {
  std::vector<std::pair<size_t, double>> obs_ids_and_times;
  const h5::H5File<h5::AccessType::ReadOnly> first_h5file(filenames.front());
  const auto& subfile = first_h5file.get<h5::VolumeData>(subfile_name);
  const auto all_observation_ids = subfile.list_observation_ids();
  if (all_observation_ids.empty()) {
    ERROR("No observation IDs found in the volume data files.");
  }
  std::vector<double> all_observation_values{};
  all_observation_values.reserve(all_observation_ids.size());
  obs_ids_and_times.reserve(all_observation_ids.size());
  for (const size_t obs_id : all_observation_ids) {
    const double obs_value = subfile.get_observation_value(obs_id);
    all_observation_values.push_back(obs_value);
    obs_ids_and_times.emplace_back(obs_id, obs_value);
  }
  first_h5file.close();
  if (std::isfinite(start_time) or std::isfinite(end_time)) {
    obs_ids_and_times.erase(
        std::remove_if(obs_ids_and_times.begin(), obs_ids_and_times.end(),
                       [start_time, end_time](const auto& id_and_time) {
                         return (std::isfinite(start_time) and
                                 id_and_time.second < start_time) or
                                (std::isfinite(end_time) and
                                 id_and_time.second > end_time);
                       }),
        obs_ids_and_times.end());
  }
  if (obs_ids_and_times.size() < 9) {
    ERROR("Need at least 9 observation times after filtering but found "
          << obs_ids_and_times.size() << ".");
  }
  const double relative_epsilon =
      100.0 * std::numeric_limits<double>::epsilon();
  const double first_time = obs_ids_and_times.front().second;
  const double uniform_time_step =
      obs_ids_and_times[1].second - obs_ids_and_times[0].second;
  if (uniform_time_step <= 0.0) {
    ERROR("Observation times for subfile '"
          << subfile_name
          << "' must be strictly increasing and uniformly spaced, but the "
             "first time step is "
          << uniform_time_step << ".");
  }
  for (size_t i = 1; i < obs_ids_and_times.size(); ++i) {
    const double actual_time = obs_ids_and_times[i].second;
    const double expected_time =
        first_time + static_cast<double>(i) * uniform_time_step;
    const double scale =
        std::max({1.0, std::abs(expected_time), std::abs(actual_time)});
    if (not equal_within_roundoff(actual_time, expected_time, relative_epsilon,
                                  scale)) {
      ERROR("Observation times for subfile '"
            << subfile_name
            << "' must be uniformly spaced, but observation index " << i
            << " has time " << actual_time << " and expected " << expected_time
            << " from first time " << first_time << " and time step "
            << uniform_time_step << ".");
    }
  }

  // Check that all other files have the same observation ids
  for (size_t file_index = 1; file_index < filenames.size(); ++file_index) {
    const h5::H5File<h5::AccessType::ReadOnly> other_h5file(
        filenames[file_index]);
    const auto& other_volfile = other_h5file.get<h5::VolumeData>(subfile_name);
    const auto other_observation_ids = other_volfile.list_observation_ids();
    if (other_observation_ids.size() != all_observation_ids.size()) {
      ERROR("Mismatched number of observation IDs between volume data files.");
    }
    for (size_t id_index = 0; id_index < all_observation_ids.size();
         ++id_index) {
      if (other_observation_ids[id_index] != all_observation_ids[id_index]) {
        ERROR("Mismatched observation IDs between volume data files.");
      }
      const double other_observation_value =
          other_volfile.get_observation_value(other_observation_ids[id_index]);
      const double reference_observation_value =
          all_observation_values[id_index];
      const double scale = std::max({1.0, std::abs(reference_observation_value),
                                     std::abs(other_observation_value)});
      if (not equal_within_roundoff(other_observation_value,
                                    reference_observation_value,
                                    relative_epsilon, scale)) {
        ERROR("Mismatched observation value for observation ID "
              << other_observation_ids[id_index] << " in file '"
              << filenames[file_index] << "'. Expected "
              << reference_observation_value << " from file '"
              << filenames.front() << "' but found " << other_observation_value
              << ".");
      }
    }
    other_h5file.close();
  }
  return obs_ids_and_times;
}

template <size_t Dim>
auto load_grids(const h5::VolumeData& volfile, const size_t obs_id) {
  const auto grid_names = volfile.get_grid_names(obs_id);
  const auto all_extents = volfile.get_extents(obs_id);
  const auto all_bases = volfile.get_bases(obs_id);
  const auto all_quadratures = volfile.get_quadratures(obs_id);
  std::vector<ElementId<Dim>> element_ids{};
  std::unordered_map<ElementId<Dim>, Mesh<Dim>> meshes{};
  element_ids.reserve(grid_names.size());
  for (const auto& grid_name : grid_names) {
    const ElementId<Dim> element_id(grid_name);
    element_ids.push_back(element_id);
    meshes[element_id] = h5::mesh_for_grid<Dim>(
        grid_name, grid_names, all_extents, all_bases, all_quadratures);
  }
  return std::make_pair(std::move(element_ids), std::move(meshes));
}

struct ModeInterpolationResult {
  boost::math::interpolators::cardinal_cubic_b_spline<double> interpolant;
  std::vector<double> node_values;
  double max_error;
  double time_step;
};

template <size_t Dim>
void enforce_legendre_basis(const Mesh<Dim>& mesh, const std::string& context) {
  const auto basis_array = mesh.basis();
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(basis_array, d) != Spectral::Basis::Legendre) {
      ERROR("ModalSpacetimeInterpolator assumes a Legendre basis, but found "
            << gsl::at(basis_array, d) << " in dimension " << d << " for "
            << context << ".");
    }
  }
}

ModeInterpolationResult interpolant_for_mode(
    const std::vector<std::pair<size_t, double>>& obs_ids_and_times,
    const std::vector<double>& values, double absolute_error) {
  const size_t total_num_observations = obs_ids_and_times.size();
  const double start_time = obs_ids_and_times.front().second;
  const double end_time = obs_ids_and_times.back().second;
  const double time_interval = end_time - start_time;
  if (not std::isfinite(time_interval) or time_interval <= 0.0) {
    ERROR("Invalid time interval for interpolant: [" << start_time << ", "
                                                     << end_time << "].");
  }
  const double initial_time_step =
      time_interval / static_cast<double>(total_num_observations - 1);
  const double high_accuracy_end_time =
      start_time +
      initial_time_step * static_cast<double>(total_num_observations - 1);
  boost::math::interpolators::cardinal_cubic_b_spline<double>
      high_accuracy_interpolant(values.begin(), values.end(), start_time,
                                initial_time_step);

  auto build_interpolant = [&high_accuracy_interpolant, start_time,
                            time_interval,
                            high_accuracy_end_time](const size_t num_points) {
    const double time_step =
        time_interval / static_cast<double>(num_points - 1);
    std::vector<double> current_values(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      const double raw_time = start_time + static_cast<double>(i) * time_step;
      double clamped_time = raw_time;
      if (clamped_time <= start_time) {
        clamped_time = std::nextafter(start_time, high_accuracy_end_time);
      } else if (clamped_time >= high_accuracy_end_time) {
        clamped_time = std::nextafter(high_accuracy_end_time, start_time);
      }
      current_values[i] = high_accuracy_interpolant(clamped_time);
    }
    return std::make_tuple(
        boost::math::interpolators::cardinal_cubic_b_spline<double>(
            current_values.begin(), current_values.end(), start_time,
            time_step),
        current_values, time_step);
  };

  auto compute_max_error = [&obs_ids_and_times, &values, start_time](
                               const auto& interpolant, const double time_step,
                               const size_t num_points) {
    const double current_end_time =
        start_time + time_step * static_cast<double>(num_points - 1);
    double max_error = 0.0;
    // leave out the last point since it should be exact and may extrapolate due
    // to roundoff error which throws an exception
    for (size_t i = 0; i < values.size() - 1; ++i) {
      const double raw_time = obs_ids_and_times[i].second;
      double clamped_time = raw_time;
      if (not std::isfinite(clamped_time)) {
        ERROR("Non-finite observation time encountered: " << clamped_time);
      }
      if (clamped_time <= start_time) {
        clamped_time = std::nextafter(start_time, current_end_time);
      } else if (clamped_time >= current_end_time) {
        clamped_time = std::nextafter(current_end_time, start_time);
      }
      const double predicted_value = interpolant(clamped_time);
      max_error = std::max(max_error, std::abs(predicted_value - values[i]));
    }
    return max_error;
  };
  size_t num_points = 6;
  auto [interpolant, selected_values, time_step] =
      build_interpolant(num_points);
  double max_error = compute_max_error(interpolant, time_step, num_points);
  while (max_error > absolute_error and num_points < total_num_observations) {
    // keep doubling number of points until we reach desired accuracy
    num_points = std::min(total_num_observations, num_points * 2);
    std::tie(interpolant, selected_values, time_step) =
        build_interpolant(num_points);
    if (num_points == total_num_observations) {
      // if we are using all points, keep error estimate from coarser grid
      break;
    }
    max_error = compute_max_error(interpolant, time_step, num_points);
  }
  return ModeInterpolationResult{
      std::move(interpolant), std::move(selected_values), max_error, time_step};
}

double estimate_error_from_coarser_interpolant(
    const std::vector<std::pair<size_t, double>>& obs_ids_and_times,
    const std::vector<double>& values) {
  const size_t total_num_observations = obs_ids_and_times.size();
  if (total_num_observations < 9) {
    ERROR(
        "Need at least 9 observations to estimate error from coarser "
        "interpolant, but found "
        << total_num_observations << ".");
  }
  const double start_time = obs_ids_and_times.front().second;
  const double end_time = obs_ids_and_times.back().second;
  const double time_interval = end_time - start_time;
  const double initial_time_step =
      time_interval / static_cast<double>(total_num_observations - 1);

  const boost::math::interpolators::cardinal_cubic_b_spline<double>
      fine_interpolant(values.begin(), values.end(), start_time,
                       initial_time_step);

  const size_t coarse_points = (total_num_observations + 1) / 2;
  std::vector<double> coarse_values;
  coarse_values.reserve(coarse_points);
  for (size_t i = 0; i < coarse_points; ++i) {
    coarse_values.push_back(values[2 * i]);
  }
  const double coarse_time_step = 2.0 * initial_time_step;
  const boost::math::interpolators::cardinal_cubic_b_spline<double>
      coarse_interpolant(coarse_values.begin(), coarse_values.end(), start_time,
                         coarse_time_step);

  double max_error = 0.0;
  const double coarse_end_time =
      start_time + coarse_time_step * static_cast<double>(coarse_points - 1);
  for (size_t i = 0; i < total_num_observations; ++i) {
    const double raw_time =
        start_time + static_cast<double>(i) * initial_time_step;
    double clamped_time = raw_time;
    if (clamped_time <= start_time) {
      clamped_time = std::nextafter(start_time, coarse_end_time);
    } else if (clamped_time >= coarse_end_time) {
      clamped_time = std::nextafter(coarse_end_time, start_time);
    }
    const double error = std::abs(fine_interpolant(clamped_time) -
                                  coarse_interpolant(clamped_time));
    max_error = std::max(max_error, error);
  }
  // for 3rd order, we would expect error to reduce by 16 when halving the step
  // size but 8 is already conservative for noisy data.
  return max_error / 8.;
}

}  // namespace

template <size_t Dim, typename Frame>
struct ModalSpacetimeInterpolator<Dim, Frame>::FileCache {
  struct ObservationCache {
    size_t obs_id{};
    std::vector<std::string> grid_names{};
    std::vector<std::vector<size_t>> extents{};
    std::vector<std::vector<Spectral::Basis>> bases{};
    std::vector<std::vector<Spectral::Quadrature>> quadratures{};
    std::unordered_map<std::string, std::pair<size_t, size_t>>
        offsets_by_grid{};
  };

  h5::H5File<h5::AccessType::ReadOnly> h5file;
  std::reference_wrapper<const h5::VolumeData> volfile;
  std::vector<ObservationCache> observations{};

  FileCache(const std::string& filename, const std::string& subfile_name,
            const std::vector<std::pair<size_t, double>>& obs_ids_and_times)
      : h5file(filename),
        volfile(std::cref(h5file.get<h5::VolumeData>(subfile_name))) {
    observations.reserve(obs_ids_and_times.size());
    for (const auto& id_and_time : obs_ids_and_times) {
      ObservationCache obs{};
      obs.obs_id = id_and_time.first;
      obs.grid_names = volfile.get().get_grid_names(obs.obs_id);
      obs.extents = volfile.get().get_extents(obs.obs_id);
      obs.bases = volfile.get().get_bases(obs.obs_id);
      obs.quadratures = volfile.get().get_quadratures(obs.obs_id);
      obs.offsets_by_grid.reserve(obs.grid_names.size());
      for (const auto& grid_name : obs.grid_names) {
        obs.offsets_by_grid.emplace(
            grid_name, h5::offset_and_length_for_grid(grid_name, obs.grid_names,
                                                      obs.extents));
      }
      observations.push_back(std::move(obs));
    }
  }
};

template <size_t Dim, typename Frame>
ModalSpacetimeInterpolator<Dim, Frame>::ModalSpacetimeInterpolator(
    std::variant<std::vector<std::string>, std::string> volume_files_or_glob,
    std::vector<std::string> subfiles_coarsest_to_finest,
    std::vector<std::string> tensor_components, const double start_time,
    const double end_time, const Verbosity verbosity)
    : volume_files_or_glob_(std::move(volume_files_or_glob)),
      tensor_components_(std::move(tensor_components)),
      verbosity_(verbosity) {
  check_boost_version();
  const auto filenames = resolve_filenames(volume_files_or_glob_);
  const std::string& last_subfile_name = subfiles_coarsest_to_finest.back();
  const auto& obs_ids_and_times =
      load_observation_ids(filenames, last_subfile_name, start_time, end_time);
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  const h5::H5File<h5::AccessType::ReadOnly> first_h5file(filenames.front());
  const auto& first_volfile =
      first_h5file.get<h5::VolumeData>(last_subfile_name);
  const size_t reference_obs_id = obs_ids_and_times.back().first;
  auto serialized_domain = first_volfile.get_domain();
  domain_ = deserialize<Domain<Dim>>(serialized_domain->data());
  auto serialized_fots = first_volfile.get_functions_of_time(reference_obs_id);
  if (serialized_fots.has_value()) {
    functions_of_time_ =
        deserialize<domain::FunctionsOfTimeMap>(serialized_fots->data());
  } else {
    functions_of_time_.clear();
  }
  gather_element_metadata(filenames, last_subfile_name, reference_obs_id);
  build_interpolators(filenames, subfiles_coarsest_to_finest, start_time,
                      end_time);
}

template <size_t Dim, typename Frame>
ModalSpacetimeInterpolator<Dim, Frame>::ModalSpacetimeInterpolator(
    const std::string& h5_filename, const std::string& group_path,
    const Verbosity verbosity)
    : volume_files_or_glob_(std::vector<std::string>{}), verbosity_(verbosity) {
  check_boost_version();
  const hid_t file_id =
      H5Fopen(h5_filename.c_str(), H5F_ACC_RDONLY, h5::h5p_default());
  CHECK_H5(file_id, "Failed to open H5 file '" << h5_filename << "'.");

  const h5::detail::OpenGroup root_group(file_id, group_path,
                                         h5::AccessType::ReadOnly);
  const auto stored_dim =
      h5::read_value_attribute<size_t>(root_group.id(), "Dimension");
  if (stored_dim != Dim) {
    ERROR("Stored dimension "
          << stored_dim << " does not match interpolator dimension " << Dim
          << ".");
  }
  const auto stored_frame =
      h5::read_value_attribute<std::string>(root_group.id(), "Frame");
  if (stored_frame != pretty_type::name<Frame>()) {
    ERROR("Stored frame '" << stored_frame
                           << "' does not match interpolator frame '"
                           << pretty_type::name<Frame>() << "'.");
  }

  const auto stored_time_bounds =
      h5::read_rank1_attribute<double>(root_group.id(), "TimeBounds");
  time_bounds_[0] = stored_time_bounds[0];
  time_bounds_[1] = stored_time_bounds[1];

  tensor_components_ = h5::read_rank1_attribute<std::string>(
      root_group.id(), "TensorComponents");

  const auto serialized_domain =
      h5::read_data<1, std::vector<char>>(root_group.id(), "Domain");
  domain_ = deserialize<Domain<Dim>>(serialized_domain.data());

  if (h5::contains_dataset_or_group(root_group.id(), "", "FunctionsOfTime")) {
    const auto serialized_fots =
        h5::read_data<1, std::vector<char>>(root_group.id(), "FunctionsOfTime");
    if (serialized_fots.empty()) {
      functions_of_time_.clear();
    } else {
      functions_of_time_ =
          deserialize<domain::FunctionsOfTimeMap>(serialized_fots.data());
    }
  } else {
    functions_of_time_.clear();
  }

  element_search_trees_.clear();
  element_data_.clear();

  if (not h5::contains_dataset_or_group(root_group.id(), "", "Elements")) {
    ERROR("Interpolator file '" << h5_filename
                                << "' does not contain an 'Elements' group.");
  }

  const h5::detail::OpenGroup elements_group(root_group.id(), "Elements",
                                             h5::AccessType::ReadOnly);
  const auto element_names = h5::get_group_names(elements_group.id(), "");
  for (const auto& element_name : element_names) {
    const h5::detail::OpenGroup element_group(elements_group.id(), element_name,
                                              h5::AccessType::ReadOnly);
    const Index<Dim> extents =
        h5::read_extents<Dim>(element_group.id(), "Extents");
    const auto basis_ints =
        h5::read_rank1_attribute<int>(element_group.id(), "Basis");
    const auto quadrature_ints =
        h5::read_rank1_attribute<int>(element_group.id(), "Quadrature");
    if (basis_ints.size() != Dim or quadrature_ints.size() != Dim) {
      ERROR("Stored basis or quadrature has incorrect size for element "
            << element_name << ".");
    }
    std::array<size_t, Dim> extents_array{};
    std::array<Spectral::Basis, Dim> bases{};
    std::array<Spectral::Quadrature, Dim> quadratures{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(extents_array, d) = extents[d];
      gsl::at(bases, d) = static_cast<Spectral::Basis>(gsl::at(basis_ints, d));
      gsl::at(quadratures, d) =
          static_cast<Spectral::Quadrature>(gsl::at(quadrature_ints, d));
    }
    const Mesh<Dim> mesh(extents_array, bases, quadratures);
    std::string mesh_context = "element ";
    mesh_context += element_name;
    mesh_context += " in '";
    mesh_context += h5_filename;
    mesh_context += "'";
    enforce_legendre_basis(mesh, mesh_context);
    ElementData element_data{};
    element_data.mesh = mesh;
    element_data.file_index = 0;
    element_data.component_interpolators.resize(tensor_components_.size());
    for (auto& component : element_data.component_interpolators) {
      component.modal_interpolants.resize(mesh.number_of_grid_points());
    }

    for (size_t component_index = 0;
         component_index < tensor_components_.size(); ++component_index) {
      const std::string component_group_name =
          "Component_" + std::to_string(component_index);
      if (not h5::contains_dataset_or_group(element_group.id(), "",
                                            component_group_name)) {
        ERROR("Missing component group '" << component_group_name
                                          << "' for element " << element_name
                                          << ".");
      }
      const h5::detail::OpenGroup component_group(
          element_group.id(), component_group_name, h5::AccessType::ReadOnly);
      const auto offsets = h5::read_data<1, std::vector<size_t>>(
          component_group.id(), "Offsets");
      const size_t num_modes = mesh.number_of_grid_points();
      if (offsets.size() != num_modes + 1) {
        ERROR("Offsets array for element "
              << element_name << " component "
              << tensor_components_[component_index] << " has size "
              << offsets.size() << ", expected " << num_modes + 1 << ".");
      }

      std::vector<double> flat_values{};
      if (h5::contains_dataset_or_group(component_group.id(), "", "Values")) {
        flat_values = h5::read_data<1, std::vector<double>>(
            component_group.id(), "Values");
      }
      const auto time_steps = h5::read_data<1, std::vector<double>>(
          component_group.id(), "TimeSteps");
      if (time_steps.size() != num_modes) {
        ERROR("TimeSteps array for element "
              << element_name << " component "
              << tensor_components_[component_index] << " has size "
              << time_steps.size() << ", expected " << num_modes << ".");
      }
      const auto start_times = h5::read_data<1, std::vector<double>>(
          component_group.id(), "StartTimes");
      if (start_times.size() != num_modes) {
        ERROR("StartTimes array for element "
              << element_name << " component "
              << tensor_components_[component_index] << " has size "
              << start_times.size() << ", expected " << num_modes << ".");
      }

      const auto has_data =
          h5::read_data<1, std::vector<int>>(component_group.id(), "HasData");
      if (has_data.size() != num_modes) {
        ERROR("HasData array for element "
              << element_name << " component "
              << tensor_components_[component_index] << " has size "
              << has_data.size() << ", expected " << num_modes << ".");
      }

      for (size_t mode_index = 0; mode_index < num_modes; ++mode_index) {
        const size_t begin = offsets[mode_index];
        const size_t end = offsets[mode_index + 1];
        auto& mode_data = element_data.component_interpolators[component_index]
                              .modal_interpolants[mode_index];
        if (has_data[mode_index] == 0 or end <= begin) {
          mode_data.interpolant.reset();
          mode_data.values.clear();
          mode_data.start_time = 0.0;
          mode_data.time_step = 0.0;
          continue;
        }
        if (end > flat_values.size()) {
          ERROR("Offset entry " << end << " exceeds stored data length "
                                << flat_values.size() << " for element "
                                << element_name << ", component "
                                << tensor_components_[component_index] << ".");
        }
        const auto begin_it =
            flat_values.begin() + static_cast<std::ptrdiff_t>(begin);
        const auto end_it =
            flat_values.begin() + static_cast<std::ptrdiff_t>(end);
        mode_data.values.assign(begin_it, end_it);
        mode_data.start_time = start_times[mode_index];
        mode_data.time_step = time_steps[mode_index];
        mode_data.interpolant =
            boost::math::interpolators::cardinal_cubic_b_spline<double>(
                mode_data.values.begin(), mode_data.values.end(),
                mode_data.start_time, mode_data.time_step);
      }
    }

    const ElementId<Dim> element_id(element_name);
    element_search_trees_[element_id.block_id()].insert(element_id);
    element_data_.emplace(element_id, std::move(element_data));
  }
  if (static_cast<int>(verbosity_) >= static_cast<int>(Verbosity::Quiet)) {
    Parallel::printf(
        "Loaded ModalSpacetimeInterpolator from '%s' with %zu elements and %zu "
        "tensor components.\n",
        h5_filename.c_str(), element_data_.size(), tensor_components_.size());
  }
}

template <size_t Dim, typename Frame>
void ModalSpacetimeInterpolator<Dim, Frame>::gather_element_metadata(
    const std::vector<std::string>& filenames, const std::string& subfile_name,
    const size_t reference_obs_id) {
  // we do a first pass over all files to gather metadata about which elements
  // exist in which files, and what their meshes are. This avoids loading all
  // data into memory at once.
  for (size_t file_index = 0; file_index < filenames.size(); ++file_index) {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filenames[file_index]);
    const auto& volfile = h5file.get<h5::VolumeData>(subfile_name);
    const auto [element_ids, meshes] =
        load_grids<Dim>(volfile, reference_obs_id);
    for (const auto& element_id : element_ids) {
      const auto& mesh = meshes.at(element_id);
      enforce_legendre_basis(mesh, "element " + get_output(element_id));
      const size_t number_of_grid_points = mesh.number_of_grid_points();
      ComponentInterpolator component_interpolator{};
      component_interpolator.modal_interpolants.resize(number_of_grid_points);
      const std::vector<ComponentInterpolator> component_interpolators(
          tensor_components_.size(), component_interpolator);
      ElementData element_data{mesh, file_index, component_interpolators};
      element_data_.emplace(element_id, std::move(element_data));
      element_search_trees_[element_id.block_id()].insert(element_id);
    }
  }
}

template <size_t Dim, typename Frame>
void ModalSpacetimeInterpolator<Dim, Frame>::build_interpolators(
    const std::vector<std::string>& filenames,
    const std::vector<std::string>& subfile_names, const double start_time,
    const double end_time) {
  // Build a time interpolant for every tensor component / grid point pair by
  // streaming the observations from disk element-by-element.
  const size_t num_components = tensor_components_.size();
  std::unordered_map<ElementId<Dim>, std::vector<double>>
      inferred_errors_by_element{};
  for (size_t i = 0; i < subfile_names.size(); ++i) {
    const std::string& subfile_name = subfile_names[i];
    const auto obs_ids_and_times =
        load_observation_ids(filenames, subfile_name, start_time, end_time);
    std::vector<FileCache> file_caches{};
    file_caches.reserve(filenames.size());
    for (const auto& filename : filenames) {
      file_caches.emplace_back(filename, subfile_name, obs_ids_and_times);
    }
    const double lower_time = obs_ids_and_times.front().second;
    const double upper_time = obs_ids_and_times.back().second;
    if (lower_time > time_bounds_[0]) {
      time_bounds_[0] = lower_time;
    }
    if (upper_time < time_bounds_[1]) {
      time_bounds_[1] = upper_time;
    }
    for (auto& [element_id, element_data] : element_data_) {
      const auto& total_mesh = element_data.mesh;
      auto& component_interpolators = element_data.component_interpolators;

      for (size_t component_index = 0; component_index < num_components;
           ++component_index) {
        auto& component_interpolator = component_interpolators[component_index];
        const auto [per_mode_values, extents] = load_component_time_series(
            file_caches.at(element_data.file_index), element_id,
            component_index, obs_ids_and_times);
        const size_t num_grid_points = per_mode_values.size();
        ASSERT(num_grid_points == extents.product(),
               "Number of grid points does not match extents.");
        auto& inferred_errors = inferred_errors_by_element[element_id];
        if (inferred_errors.empty()) {
          inferred_errors.resize(num_components,
                                 std::numeric_limits<double>::max());
        }
        // estimate error from (0,0,0) mode from the first subfile
        if (i == 0) {
          Index<Dim> zero_index{};
          for (size_t d = 0; d < Dim; ++d) {
            zero_index[d] = 0;
          }
          const size_t zero_mode_index =
              collapsed_index<Dim>(zero_index, extents);
          inferred_errors[component_index] =
              estimate_error_from_coarser_interpolant(
                  obs_ids_and_times, per_mode_values[zero_mode_index]);
          if (static_cast<int>(verbosity_) >=
              static_cast<int>(Verbosity::Verbose)) {
            Parallel::printf(
                "Absolute error target for element %s, component %s: %.3e.\n",
                get_output(element_id).c_str(),
                tensor_components_[component_index].c_str(),
                inferred_errors[component_index]);
          }
        }
        // clamp error in case of very small or zero error
        const double mode_absolute_error_tolerance =
            std::max(inferred_errors[component_index], 1.0e-16);
        for (size_t mode_index = 0; mode_index < num_grid_points;
             ++mode_index) {
          const auto full_index = expanded_index<Dim>(mode_index, extents);
          const auto collapsed_total_index =
              collapsed_index<Dim>(full_index, total_mesh.extents());
          auto& mode_data = component_interpolator.modal_interpolants.at(
              collapsed_total_index);
          // if we already have an interpolant for that mode from a previous
          // subfile, use that one as subfiles are orderer from coarsest to
          // finest
          if (mode_data.interpolant.has_value()) {
            continue;
          }
          const auto& values = per_mode_values[mode_index];
          const auto max_abs_it = std::max_element(
              values.begin(), values.end(), [](double lhs, double rhs) {
                return std::abs(lhs) < std::abs(rhs);
              });
          const double max_abs_value =
              max_abs_it == values.end() ? 0.0 : std::abs(*max_abs_it);
          if (max_abs_value <= mode_absolute_error_tolerance) {
            mode_data.interpolant.reset();
            mode_data.values.clear();
            mode_data.start_time = 0.0;
            mode_data.time_step = 0.0;
            continue;
          }
          auto interpolation_result = interpolant_for_mode(
              obs_ids_and_times, values, mode_absolute_error_tolerance);
          if (static_cast<int>(verbosity_) >=
              static_cast<int>(Verbosity::Verbose)) {
            const auto full_index_str = get_output(full_index);
            if (interpolation_result.max_error >
                mode_absolute_error_tolerance) {
              Parallel::printf(
                  "For element %s, component %s, mode %s, could not achieve "
                  "the requested error tolerance %.3e; achieved "
                  "maximum error %.3e using time step %.3e.\n",
                  get_output(element_id).c_str(),
                  tensor_components_[component_index].c_str(),
                  full_index_str.c_str(), mode_absolute_error_tolerance,
                  interpolation_result.max_error,
                  interpolation_result.time_step);
            } else {
              Parallel::printf(
                  "For element %s, component %s, mode %s, achieved "
                  "maximum error %.3e using time step %.3e.\n",
                  get_output(element_id).c_str(),
                  tensor_components_[component_index].c_str(),
                  full_index_str.c_str(), interpolation_result.max_error,
                  interpolation_result.time_step);
            }
          }
          mode_data.start_time = obs_ids_and_times.front().second;
          mode_data.time_step = interpolation_result.time_step;
          mode_data.values = std::move(interpolation_result.node_values);
          mode_data.interpolant = std::move(interpolation_result.interpolant);
        }
      }
      if (static_cast<int>(verbosity_) >= static_cast<int>(Verbosity::Quiet)) {
        Parallel::printf(
            "Constructed interpolator for element %s with %zu tensor "
            "components.\n",
            get_output(element_id).c_str(), num_components);
      }
    }
  }
}

template <size_t Dim, typename Frame>
std::pair<std::vector<std::vector<double>>, Index<Dim>>
ModalSpacetimeInterpolator<Dim, Frame>::load_component_time_series(
    const FileCache& file_cache, const ElementId<Dim>& element_id,
    const size_t component_index,
    const std::vector<std::pair<size_t, double>>& obs_ids_and_times) const {
  const size_t num_observations = obs_ids_and_times.size();
  const std::string element_name = get_output(element_id);
  ASSERT(file_cache.observations.size() == num_observations,
         "Cached observations do not match requested observation count.");
  const auto& element_volfile = file_cache.volfile.get();
  const auto& first_obs_cache = file_cache.observations.front();
  const auto& all_grid_names_first_obs = first_obs_cache.grid_names;
  const auto& all_extents_first_obs = first_obs_cache.extents;
  const auto& all_bases_first_obs = first_obs_cache.bases;
  const auto& all_quadratures_first_obs = first_obs_cache.quadratures;
  const auto mesh_first_obs = h5::mesh_for_grid<Dim>(
      element_name, all_grid_names_first_obs, all_extents_first_obs,
      all_bases_first_obs, all_quadratures_first_obs);
  enforce_legendre_basis(mesh_first_obs,
                         "element " + element_name + " at first observation");
  const size_t num_grid_points = mesh_first_obs.number_of_grid_points();
  std::vector<std::vector<double>> per_mode_values(
      num_grid_points, std::vector<double>(num_observations, 0.0));
  DataVector nodal_data(num_grid_points, 0.0);
  ModalVector modal_data(num_grid_points, 0.0);
  for (size_t obs_index = 0; obs_index < num_observations; ++obs_index) {
    const auto& obs_cache = file_cache.observations[obs_index];
    const size_t obs_id = obs_cache.obs_id;
    const auto& all_grid_names = obs_cache.grid_names;
    const auto& all_extents = obs_cache.extents;
    const auto& all_bases = obs_cache.bases;
    const auto& all_quadratures = obs_cache.quadratures;
    const auto mesh = h5::mesh_for_grid<Dim>(
        element_name, all_grid_names, all_extents, all_bases, all_quadratures);
    const auto offset_it = obs_cache.offsets_by_grid.find(element_name);
    ASSERT(offset_it != obs_cache.offsets_by_grid.end(),
           "Element " << element_id << " is not present "
                      << " for observation " << obs_id
                      << ". Each element is expected to reside in the same "
                         "volume file for all observations.");
    const auto [offset, length] = offset_it->second;
    ASSERT(length == num_grid_points,
           "Expected " << num_grid_points << " grid points for element "
                       << element_id << ", but observation " << obs_id
                       << " provides " << length << ".");

    if (mesh != mesh_first_obs) {
      ERROR("Element " << element_id
                       << " has inconsistent mesh between observations. AMR is "
                          "not yet supported");
    }
    const auto tensor_component = element_volfile.get_tensor_component(
        obs_id, tensor_components_[component_index]);
    const auto& component_data = tensor_component.data;
    if (std::holds_alternative<DataVector>(component_data)) {
      const auto& data = std::get<DataVector>(component_data);
      std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(offset),
                  static_cast<std::ptrdiff_t>(num_grid_points),
                  nodal_data.begin());
    } else {
      const auto& data = std::get<std::vector<float>>(component_data);
      std::transform(data.begin() + static_cast<std::ptrdiff_t>(offset),
                     data.begin() + static_cast<std::ptrdiff_t>(offset) +
                         static_cast<std::ptrdiff_t>(num_grid_points),
                     nodal_data.begin(), [](const float value) {
                       return static_cast<double>(value);
                     });
    }
    to_modal_coefficients(make_not_null(&modal_data), nodal_data, mesh);
    for (size_t mode = 0; mode < num_grid_points; ++mode) {
      per_mode_values[mode][obs_index] = modal_data[mode];
    }
  }
  return std::make_pair(per_mode_values, mesh_first_obs.extents());
}

template <size_t Dim, typename Frame>
void ModalSpacetimeInterpolator<Dim, Frame>::write_to_h5(
    const std::string& h5_filename, const std::string& group_path) const {
  if (UNLIKELY(element_data_.empty())) {
    ERROR(
        "Cannot write ModalSpacetimeInterpolator to H5 file because no "
        "element data is available.");
  }

  const hid_t file_id = H5Fcreate(h5_filename.c_str(), H5F_ACC_TRUNC,
                                  h5::h5p_default(), h5::h5p_default());
  CHECK_H5(file_id, "Failed to create H5 file '" << h5_filename << "'.");

  const h5::detail::OpenGroup root_group(file_id, group_path,
                                         h5::AccessType::ReadWrite);

  h5::write_to_attribute(root_group.id(), "Dimension", Dim);
  h5::write_to_attribute(root_group.id(), "Frame", pretty_type::name<Frame>());
  h5::write_to_attribute(root_group.id(), "TimeBounds",
                         std::vector<double>{time_bounds_[0], time_bounds_[1]});
  h5::write_to_attribute(root_group.id(), "TensorComponents",
                         tensor_components_);

  const auto serialized_domain = serialize<Domain<Dim>>(domain_);
  h5::write_data(root_group.id(), serialized_domain, {serialized_domain.size()},
                 "Domain");

  const auto serialized_fots = serialize(functions_of_time_);
  if (not serialized_fots.empty()) {
    h5::write_data(root_group.id(), serialized_fots, {serialized_fots.size()},
                   "FunctionsOfTime");
  }

  const h5::detail::OpenGroup elements_group(root_group.id(), "Elements",
                                             h5::AccessType::ReadWrite);
  for (const auto& [element_id, data] : element_data_) {
    const std::string element_name = get_output(element_id);
    const h5::detail::OpenGroup element_group(elements_group.id(), element_name,
                                              h5::AccessType::ReadWrite);
    h5::write_extents<Dim>(element_group.id(), data.mesh.extents());
    const auto bases = data.mesh.basis();
    const auto quadratures = data.mesh.quadrature();
    std::vector<int> basis_ints(Dim);
    std::vector<int> quadrature_ints(Dim);
    for (size_t d = 0; d < Dim; ++d) {
      basis_ints.at(d) = static_cast<int>(gsl::at(bases, d));
      quadrature_ints.at(d) = static_cast<int>(gsl::at(quadratures, d));
    }
    h5::write_to_attribute(element_group.id(), "Basis", basis_ints);
    h5::write_to_attribute(element_group.id(), "Quadrature", quadrature_ints);

    for (size_t component_index = 0;
         component_index < tensor_components_.size(); ++component_index) {
      const std::string component_group_name =
          "Component_" + std::to_string(component_index);
      const h5::detail::OpenGroup component_group(
          element_group.id(), component_group_name, h5::AccessType::ReadWrite);
      const auto& component_interpolator =
          data.component_interpolators[component_index];
      const size_t num_modes = component_interpolator.modal_interpolants.size();
      if (num_modes == 0) {
        continue;
      }

      std::vector<size_t> offsets;
      offsets.reserve(num_modes + 1);
      offsets.push_back(0);
      std::vector<double> values_flat{};
      std::vector<double> time_steps(
          num_modes, std::numeric_limits<double>::signaling_NaN());
      std::vector<double> start_times(
          num_modes, std::numeric_limits<double>::signaling_NaN());
      std::vector<int> has_data(num_modes, 0);

      for (size_t mode_index = 0; mode_index < num_modes; ++mode_index) {
        const auto& mode_data =
            component_interpolator.modal_interpolants[mode_index];
        if (mode_data.interpolant.has_value()) {
          has_data[mode_index] = 1;
          values_flat.insert(values_flat.end(), mode_data.values.begin(),
                             mode_data.values.end());
          time_steps[mode_index] = mode_data.time_step;
          start_times[mode_index] = mode_data.start_time;
        }
        offsets.push_back(values_flat.size());
      }

      if (not values_flat.empty()) {
        h5::write_data(component_group.id(), values_flat, {values_flat.size()},
                       "Values");
      }
      h5::write_data(component_group.id(), offsets, {offsets.size()},
                     "Offsets");
      h5::write_data(component_group.id(), has_data, {has_data.size()},
                     "HasData");
      h5::write_data(component_group.id(), time_steps, {time_steps.size()},
                     "TimeSteps");
      h5::write_data(component_group.id(), start_times, {start_times.size()},
                     "StartTimes");
    }
  }
}

template <size_t Dim, typename Frame>
void ModalSpacetimeInterpolator<Dim, Frame>::interpolate_to_point(
    const gsl::not_null<std::vector<double>*> result,
    const tnsr::I<double, Dim, Frame>& target_point, const double time,
    const std::optional<gsl::not_null<std::vector<size_t>*>> block_order)
    const {
  ASSERT(time >= time_bounds_[0] and time <= time_bounds_[1],
         "Requested time " << time
                           << " lies outside the available data interval "
                           << time_bounds_ << ".");
  const auto block_logical_coords = block_logical_coordinates_single_point(
      target_point, domain_, time, functions_of_time_, block_order);
  ASSERT(block_logical_coords.has_value(), "Point is not in any block:\n"
                                               << target_point);

  const auto& block_pair = block_logical_coords.value();
  const auto element_coords =
      element_logical_coordinates(block_pair, element_search_trees_);
  ASSERT(element_coords.has_value(),
         "Failed to determine element logical coordinates for point "
             << target_point << " at time " << time << ".");

  const auto& element_id = element_coords->first;
  const auto& logical_coords = element_coords->second;

  const auto& element_data = element_data_.at(element_id);
  const auto& component_interpolators = element_data.component_interpolators;
  const Mesh<Dim>& mesh = element_data.mesh;
  const size_t num_components = component_interpolators.size();
  ASSERT(num_components == tensor_components_.size(),
         "Inconsistent number of tensor components stored in interpolator.");

  const size_t num_grid_points = mesh.number_of_grid_points();
  ModalVector modal_values(num_grid_points);

  result->resize(num_components);
  for (size_t component_index = 0; component_index < num_components;
       ++component_index) {
    const auto& component_interpolator =
        component_interpolators[component_index];
    ASSERT(component_interpolator.modal_interpolants.size() == num_grid_points,
           "Stored modal interpolants do not match mesh size for element "
               << element_id << ".");

    for (size_t point = 0; point < num_grid_points; ++point) {
      const auto& mode_data = component_interpolator.modal_interpolants[point];
      if (not mode_data.interpolant.has_value()) {
        modal_values[point] = 0.0;
        continue;
      }
      modal_values[point] = mode_data.interpolant.value()(time);
    }
    (*result)[component_index] = Spectral::evaluate_legendre_series<Dim>(
        modal_values, mesh, logical_coords);
  }
}

// Explicit instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class ModalSpacetimeInterpolator<DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial))

#undef INSTANTIATE
#undef DIM
#undef FRAME

}  // namespace spectre::Exporter
