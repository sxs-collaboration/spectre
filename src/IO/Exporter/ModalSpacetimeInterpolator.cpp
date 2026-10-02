// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/ModalSpacetimeInterpolator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "IO/Exporter/ModalTimeSeriesReader.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Legendre.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

namespace {

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
    ERROR_NO_TRACE("No volume files found. Specify at least one volume file.");
  }
  return filenames;
}

void validate_requested_time_bounds(const std::optional<double> start_time,
                                    const std::optional<double> end_time) {
  if (start_time.has_value() and not std::isfinite(*start_time)) {
    ERROR_NO_TRACE("The requested start time must be finite, but is "
                   << *start_time << ".");
  }
  if (end_time.has_value() and not std::isfinite(*end_time)) {
    ERROR_NO_TRACE("The requested end time must be finite, but is " << *end_time
                                                                    << ".");
  }
  if (start_time.has_value() and end_time.has_value() and
      *start_time > *end_time) {
    ERROR_NO_TRACE("The requested start time "
                   << *start_time << " is after the requested end time "
                   << *end_time << ".");
  }
}

void validate_unique_names(const std::vector<std::string>& names,
                           const std::string& description) {
  const std::unordered_set<std::string> unique_names{names.begin(),
                                                     names.end()};
  if (unique_names.size() != names.size()) {
    ERROR_NO_TRACE(description << " must be unique, but received " << names
                               << ".");
  }
}

template <size_t Dim>
std::unordered_map<ElementId<Dim>, Mesh<Dim>> element_meshes(
    const ModalTimeSeriesReader<Dim>& reader) {
  std::unordered_map<ElementId<Dim>, Mesh<Dim>> result{};
  result.reserve(reader.elements().size());
  for (const auto& [element_id, mesh] : reader.elements()) {
    result.emplace(element_id, mesh);
  }
  return result;
}

template <size_t Dim>
void validate_nested_meshes(
    const std::vector<std::unordered_map<ElementId<Dim>, Mesh<Dim>>>&
        meshes_by_subfile,
    const std::vector<std::string>& subfile_names) {
  const auto& final_meshes = meshes_by_subfile.back();
  for (size_t subfile_index = 0; subfile_index + 1 < meshes_by_subfile.size();
       ++subfile_index) {
    const auto& current_meshes = meshes_by_subfile[subfile_index];
    const auto& next_meshes = meshes_by_subfile[subfile_index + 1];
    if (current_meshes.size() != final_meshes.size()) {
      ERROR_NO_TRACE("Subfile '"
                     << subfile_names[subfile_index] << "' contains "
                     << current_meshes.size() << " elements, but the final "
                     << "subfile '" << subfile_names.back() << "' contains "
                     << final_meshes.size()
                     << ". All subfiles must contain the same elements.");
    }
    for (const auto& [element_id, final_mesh] : final_meshes) {
      const auto current_it = current_meshes.find(element_id);
      if (current_it == current_meshes.end()) {
        ERROR_NO_TRACE("Element "
                       << element_id << " is missing from subfile '"
                       << subfile_names[subfile_index]
                       << "'. All subfiles must contain the same elements.");
      }
      const auto next_it = next_meshes.find(element_id);
      if (next_it == next_meshes.end()) {
        ERROR_NO_TRACE("Element "
                       << element_id << " is missing from subfile '"
                       << subfile_names[subfile_index + 1]
                       << "'. All subfiles must contain the same elements.");
      }
      const auto& current_extents = current_it->second.extents();
      const auto& next_extents = next_it->second.extents();
      for (size_t d = 0; d < Dim; ++d) {
        if (current_extents[d] > next_extents[d]) {
          ERROR_NO_TRACE("The mesh for element "
                         << element_id << " has extent " << current_extents[d]
                         << " in dimension " << d << " in subfile '"
                         << subfile_names[subfile_index]
                         << "', but the next-priority subfile '"
                         << subfile_names[subfile_index + 1] << "' has extent "
                         << next_extents[d]
                         << ". Subfile meshes must have component-wise "
                            "nondecreasing extents.");
        }
      }
    }
  }
}

template <size_t Dim>
void validate_functions_of_time(
    const Domain<Dim>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const std::array<double, 2>& time_bounds) {
  std::unordered_set<std::string> required_names{};
  for (const auto& block : domain.blocks()) {
    if (block.is_time_dependent()) {
      const auto& block_names =
          block.moving_mesh_grid_to_inertial_map().function_of_time_names();
      required_names.insert(block_names.begin(), block_names.end());
    }
  }
  for (const auto& name : required_names) {
    const auto function_it = functions_of_time.find(name);
    if (function_it == functions_of_time.end()) {
      ERROR_NO_TRACE("The domain requires the function of time '"
                     << name
                     << "', but it is absent from the global functions of "
                        "time stored in the volume data.");
    }
    const auto function_bounds = function_it->second->time_bounds();
    if (function_bounds[0] > time_bounds[0] or
        function_bounds[1] < time_bounds[1]) {
      ERROR_NO_TRACE("The function of time '"
                     << name << "' is valid on " << function_bounds
                     << ", which does not cover the interpolator time "
                        "interval "
                     << time_bounds << ".");
    }
  }
}

template <size_t Dim>
void validate_series(const typename ModalTimeSeriesReader<Dim>::Series& series,
                     const Mesh<Dim>& mesh, const size_t num_components,
                     const size_t num_observations,
                     const ElementId<Dim>& element_id,
                     const std::vector<std::string>& tensor_components,
                     const std::string& subfile_name) {
  ASSERT(series.size() == num_components,
         "ModalTimeSeriesReader returned "
             << series.size() << " components, but " << num_components
             << " were requested.");
  const size_t num_modes = mesh.number_of_grid_points();
  for (size_t component_index = 0; component_index < num_components;
       ++component_index) {
    const auto& component_series = series[component_index];
    ASSERT(component_series.size() == num_modes,
           "ModalTimeSeriesReader returned "
               << component_series.size() << " modes for element " << element_id
               << ", but its mesh has " << num_modes << " points.");
    for (size_t mode = 0; mode < num_modes; ++mode) {
      const auto& values = component_series[mode];
      ASSERT(values.size() == num_observations,
             "ModalTimeSeriesReader returned "
                 << values.size() << " observations for element " << element_id
                 << ", but " << num_observations << " were expected.");
      if (const auto non_finite_it = alg::find_if(
              values,
              [](const double value) { return not std::isfinite(value); });
          non_finite_it != values.end()) {
        ERROR_NO_TRACE("Non-finite modal data found for element "
                       << element_id << ", tensor component '"
                       << tensor_components[component_index] << "', mode "
                       << mode << " in subfile '" << subfile_name << "'.");
      }
    }
  }
}

}  // namespace

template <size_t Dim, typename Frame>
ModalSpacetimeInterpolator<Dim, Frame>::ModalSpacetimeInterpolator(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    std::vector<std::string> subfiles_in_priority_order,
    std::vector<std::string> tensor_components,
    const std::optional<double> start_time,
    const std::optional<double> end_time,
    const std::optional<double> error_tolerance, const Verbosity verbosity)
    : tensor_components_(std::move(tensor_components)),
      time_bounds_{{-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()}} {
  validate_requested_time_bounds(start_time, end_time);
  if (error_tolerance.has_value() and
      (not std::isfinite(*error_tolerance) or *error_tolerance < 0.0)) {
    ERROR_NO_TRACE(
        "The error tolerance must be finite and non-negative, but is "
        << *error_tolerance << ".");
  }
  if (subfiles_in_priority_order.empty()) {
    ERROR_NO_TRACE("Specify at least one volume subfile.");
  }
  if (tensor_components_.empty()) {
    ERROR_NO_TRACE("Specify at least one tensor component.");
  }
  validate_unique_names(subfiles_in_priority_order, "Volume subfile names");
  validate_unique_names(tensor_components_, "Tensor component names");

  const std::vector<std::string> filenames =
      resolve_filenames(volume_files_or_glob);
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  std::vector<ModalTimeSeriesReader<Dim>> readers{};
  readers.reserve(subfiles_in_priority_order.size());
  for (const auto& subfile_name : subfiles_in_priority_order) {
    readers.emplace_back(filenames, subfile_name, tensor_components_,
                         start_time, end_time);
  }
  if (not error_tolerance.has_value() and
      readers.front().num_observations() < 9) {
    ERROR_NO_TRACE(
        "At least 9 observations are required in the first "
        "subfile to estimate interpolation errors, but subfile '"
        << subfiles_in_priority_order.front() << "' contains "
        << readers.front().num_observations()
        << ". Specify an error tolerance to skip the estimate.");
  }
  for (size_t subfile_index = 0; subfile_index < readers.size();
       ++subfile_index) {
    const auto& reader = readers[subfile_index];
    if (reader.num_observations() < 5) {
      ERROR_NO_TRACE(
          "At least 5 observations are required to construct "
          "cubic B-splines, but subfile '"
          << subfiles_in_priority_order[subfile_index] << "' contains "
          << reader.num_observations() << ".");
    }
    const double reader_end_time =
        reader.start_time() +
        static_cast<double>(reader.num_observations() - 1) * reader.time_step();
    time_bounds_[0] = std::max(time_bounds_[0], reader.start_time());
    time_bounds_[1] = std::min(time_bounds_[1], reader_end_time);
  }
  if (time_bounds_[0] > time_bounds_[1]) {
    ERROR_NO_TRACE(
        "The volume subfiles do not have a common time interval. "
        "Their intersection is "
        << time_bounds_ << ".");
  }

  // The finest subfile defines the domain used for point location. Check that
  // the other subfiles describe the same domain.
  {
    const h5::H5File<h5::AccessType::ReadOnly> finest_h5file(filenames.front());
    const auto& finest_volfile =
        finest_h5file.get<h5::VolumeData>(subfiles_in_priority_order.back());
    const auto serialized_domain = finest_volfile.get_domain();
    if (not serialized_domain.has_value()) {
      ERROR_NO_TRACE("Subfile '" << subfiles_in_priority_order.back()
                                 << "' in file '" << filenames.front()
                                 << "' does not contain a serialized domain.");
    }
    domain_ = deserialize<Domain<Dim>>(serialized_domain->data());
    if (const auto serialized_functions_of_time =
            finest_volfile.get_global_functions_of_time();
        serialized_functions_of_time.has_value()) {
      functions_of_time_ = deserialize<domain::FunctionsOfTimeMap>(
          serialized_functions_of_time->data());
    }
  }
  for (size_t subfile_index = 0;
       subfile_index + 1 < subfiles_in_priority_order.size(); ++subfile_index) {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filenames.front());
    const auto& volfile =
        h5file.get<h5::VolumeData>(subfiles_in_priority_order[subfile_index]);
    const auto serialized_domain = volfile.get_domain();
    if (not serialized_domain.has_value()) {
      ERROR_NO_TRACE("Subfile '" << subfiles_in_priority_order[subfile_index]
                                 << "' in file '" << filenames.front()
                                 << "' does not contain a serialized domain.");
    }
    const auto other_domain =
        deserialize<Domain<Dim>>(serialized_domain->data());
    if (other_domain != domain_) {
      ERROR_NO_TRACE("Subfiles '"
                     << subfiles_in_priority_order[subfile_index] << "' and '"
                     << subfiles_in_priority_order.back()
                     << "' contain different domains. Domain changes within "
                        "one modal spacetime interpolator are not supported.");
    }
  }
  validate_functions_of_time(domain_, functions_of_time_, time_bounds_);

  std::vector<std::unordered_map<ElementId<Dim>, Mesh<Dim>>>
      meshes_by_subfile{};
  meshes_by_subfile.reserve(readers.size());
  for (const auto& reader : readers) {
    meshes_by_subfile.push_back(element_meshes(reader));
  }
  validate_nested_meshes(meshes_by_subfile, subfiles_in_priority_order);

  const size_t num_components = tensor_components_.size();
  std::vector<ElementId<Dim>> element_ids{};
  element_ids.reserve(readers.back().elements().size());
  if (readers.back().elements().empty()) {
    ERROR_NO_TRACE("The final volume subfile contains no elements.");
  }
  for (const auto& [element_id, mesh] : readers.back().elements()) {
    if (element_id.block_id() >= domain_.blocks().size()) {
      ERROR_NO_TRACE("Element " << element_id << " refers to block "
                                << element_id.block_id()
                                << ", but the domain has only "
                                << domain_.blocks().size() << " blocks.");
    }
    ElementData element_data{};
    element_data.mesh = mesh;
    element_data.interpolants.resize(
        num_components,
        std::vector<ModeInterpolant>(mesh.number_of_grid_points()));
    element_data_.emplace(element_id, std::move(element_data));
    element_ids.push_back(element_id);
  }
  element_search_trees_ = domain::index_element_ids(element_ids);

  std::unordered_map<ElementId<Dim>, std::vector<std::vector<bool>>>
      assigned_modes{};
  std::unordered_map<ElementId<Dim>, std::vector<double>> error_tolerances{};
  assigned_modes.reserve(element_data_.size());
  error_tolerances.reserve(element_data_.size());
  for (const auto& [element_id, element_data] : element_data_) {
    assigned_modes.emplace(
        element_id, std::vector<std::vector<bool>>(
                        num_components,
                        std::vector<bool>(
                            element_data.mesh.number_of_grid_points(), false)));
    error_tolerances.emplace(
        element_id,
        std::vector<double>(num_components,
                            error_tolerance.value_or(
                                std::numeric_limits<double>::signaling_NaN())));
  }

  size_t retained_modes = 0;
  size_t dropped_modes = 0;
  size_t stored_samples = 0;
  size_t source_samples = 0;
  for (size_t subfile_index = 0; subfile_index < readers.size();
       ++subfile_index) {
    auto& reader = readers[subfile_index];
    for (const auto& [element_id, source_mesh] : reader.elements()) {
      const auto series = reader.modal_time_series(element_id);
      validate_series(series, source_mesh, num_components,
                      reader.num_observations(), element_id, tensor_components_,
                      subfiles_in_priority_order[subfile_index]);
      auto& tolerances = error_tolerances.at(element_id);
      if (subfile_index == 0 and not error_tolerance.has_value()) {
        for (size_t component_index = 0; component_index < num_components;
             ++component_index) {
          tolerances[component_index] = intrp::estimate_interpolation_error(
              series[component_index][0], reader.start_time(),
              reader.time_step());
        }
      }

      auto& destination = element_data_.at(element_id);
      auto& element_assigned_modes = assigned_modes.at(element_id);
      const auto& source_extents = source_mesh.extents();
      const auto& destination_extents = destination.mesh.extents();
      const size_t num_source_modes = source_mesh.number_of_grid_points();
      for (size_t component_index = 0; component_index < num_components;
           ++component_index) {
        const double tolerance = tolerances[component_index];
        if (not std::isfinite(tolerance) or tolerance < 0.0) {
          ERROR_NO_TRACE("Invalid error tolerance "
                         << tolerance << " for element " << element_id
                         << " and component '"
                         << tensor_components_[component_index]
                         << "'. The modal data may be too large to estimate "
                            "an interpolation error safely.");
        }
        for (size_t source_mode = 0; source_mode < num_source_modes;
             ++source_mode) {
          const auto multi_index =
              expanded_index<Dim>(source_mode, source_extents);
          const size_t destination_mode =
              collapsed_index<Dim>(multi_index, destination_extents);
          if (element_assigned_modes[component_index][destination_mode]) {
            continue;
          }
          element_assigned_modes[component_index][destination_mode] = true;
          const auto& values = series[component_index][source_mode];
          const auto max_abs_it = std::ranges::max_element(
              values, [](const double lhs, const double rhs) {
                return std::abs(lhs) < std::abs(rhs);
              });
          ASSERT(max_abs_it != values.end(),
                 "Modal time series unexpectedly contains no values.");
          if (std::abs(*max_abs_it) <= tolerance) {
            ++dropped_modes;
            continue;
          }
          source_samples += values.size();
          auto [interpolant, last_coarser_error] = intrp::compress_to_tolerance(
              values, reader.start_time(), reader.time_step(), tolerance);
          stored_samples += interpolant.values().size();
          destination.interpolants[component_index][destination_mode] =
              std::move(interpolant);
          ++retained_modes;
          if (verbosity >= Verbosity::Debug and
              last_coarser_error > tolerance) {
            Parallel::printf(
                "For element %s, component %s, mode %zu, retained all "
                "samples; the last coarser candidate had error %.3e above "
                "the tolerance %.3e.\n",
                get_output(element_id).c_str(),
                tensor_components_[component_index].c_str(), destination_mode,
                last_coarser_error, tolerance);
          }
        }
      }
      if (verbosity >= Verbosity::Verbose) {
        Parallel::printf("Processed element %s from subfile %s.\n",
                         get_output(element_id).c_str(),
                         subfiles_in_priority_order[subfile_index].c_str());
      }
    }
  }

  for (const auto& [element_id, component_modes] : assigned_modes) {
    for (size_t component_index = 0; component_index < num_components;
         ++component_index) {
      if (alg::find(component_modes[component_index], false) !=
          component_modes[component_index].end()) {
        ERROR_NO_TRACE("Not all modes were assigned for element "
                       << element_id << ", tensor component '"
                       << tensor_components_[component_index]
                       << "'. Check that the subfile meshes form a nested "
                          "hierarchy ending at the final mesh.");
      }
    }
  }

  if (verbosity >= Verbosity::Quiet) {
    const double compression_ratio =
        source_samples == 0 ? 0.0
                            : static_cast<double>(stored_samples) /
                                  static_cast<double>(source_samples);
    Parallel::printf(
        "Constructed ModalSpacetimeInterpolator with %zu elements, %zu "
        "components, and time bounds [%.16g, %.16g]. Retained %zu modes, "
        "dropped %zu modes, and stored %.3f of the retained source samples.\n",
        element_data_.size(), num_components, time_bounds_[0], time_bounds_[1],
        retained_modes, dropped_modes, compression_ratio);
  }
}

template <size_t Dim, typename Frame>
void ModalSpacetimeInterpolator<Dim, Frame>::interpolate_to_point(
    const gsl::not_null<std::vector<double>*> result,
    const tnsr::I<double, Dim, Frame>& target_point, const double time,
    const std::optional<gsl::not_null<std::vector<size_t>*>> block_order)
    const {
  if (not std::isfinite(time) or time < time_bounds_[0] or
      time > time_bounds_[1]) {
    ERROR_NO_TRACE("Requested time "
                   << time << " lies outside the available data interval "
                   << time_bounds_ << ".");
  }
  const auto block_logical_coords = block_logical_coordinates_single_point(
      target_point, domain_, time, functions_of_time_, block_order);
  if (not block_logical_coords.has_value()) {
    ERROR_NO_TRACE("Point is not in any block:\n" << target_point);
  }
  const auto element_coords = element_logical_coordinates(
      block_logical_coords.value(), element_search_trees_);
  if (not element_coords.has_value()) {
    ERROR_NO_TRACE("Failed to determine element-logical coordinates for point "
                   << target_point << " at time " << time << ".");
  }

  const auto& [element_id, logical_coords] = element_coords.value();
  const auto& element_data = element_data_.at(element_id);
  const size_t num_components = tensor_components_.size();
  const size_t num_modes = element_data.mesh.number_of_grid_points();
  ModalVector modal_values(num_modes);
  result->resize(num_components);
  for (size_t component_index = 0; component_index < num_components;
       ++component_index) {
    const auto& interpolants = element_data.interpolants[component_index];
    ASSERT(interpolants.size() == num_modes,
           "Stored modal interpolants do not match the element mesh.");
    for (size_t mode = 0; mode < num_modes; ++mode) {
      modal_values[mode] =
          interpolants[mode].has_value() ? (*interpolants[mode])(time) : 0.0;
    }
    (*result)[component_index] = Spectral::evaluate_legendre_series<Dim>(
        modal_values, element_data.mesh, logical_coords);
  }
}

// Explicit instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class ModalSpacetimeInterpolator<DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
#undef DIM

}  // namespace spectre::Exporter
