// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/Exporter.hpp"

#include <csignal>  // For Blaze error handling without PCH
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Serialization/Serialize.hpp"

// Ignore OpenMP pragmas when OpenMP is not enabled
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"

namespace spectre::Exporter {

namespace {
template <size_t Dim>
void interpolate_to_points(
    const gsl::not_null<std::vector<std::vector<double>>*> result,
    const gsl::not_null<std::vector<bool>*> filled_data,
    const std::string& filename, const std::string& subfile_name,
    const size_t obs_id, const std::vector<std::string>& tensor_components,
    const std::vector<std::optional<
        IdPair<domain::BlockId, tnsr::I<double, Dim, Frame::BlockLogical>>>>&
        block_logical_coords) {
  std::vector<std::string> grid_names;
  std::vector<std::vector<size_t>> all_extents;
  std::vector<std::vector<Spectral::Basis>> all_bases;
  std::vector<std::vector<Spectral::Quadrature>> all_quadratures;
  std::vector<DataVector> tensor_data{};
  tensor_data.reserve(tensor_components.size());
  // HDF5 is not generally thread-safe, so we make sure only one thread is doing
  // HDF5 operations at a time
#pragma omp critical
  {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
    const auto& volfile = h5file.get<h5::VolumeData>(subfile_name);
    grid_names = volfile.get_grid_names(obs_id);
    all_extents = volfile.get_extents(obs_id);
    all_bases = volfile.get_bases(obs_id);
    all_quadratures = volfile.get_quadratures(obs_id);
    // Load the tensor data for all grids in the file because it's stored
    // contiguously
    for (const auto& tensor_component : tensor_components) {
      auto component_data =
          volfile.get_tensor_component(obs_id, tensor_component).data;
      if (std::holds_alternative<DataVector>(component_data)) {
        tensor_data.push_back(std::get<DataVector>(std::move(component_data)));
      } else {
        // Possible optimization: do single-precision interpolation if the
        // volume data is single-precision
        const auto& float_component_data =
            std::get<std::vector<float>>(component_data);
        DataVector double_component_data(float_component_data.size());
        for (size_t i = 0; i < float_component_data.size(); ++i) {
          double_component_data[i] = float_component_data[i];
        }
      }
    }
  }
  // Reconstruct element IDs & meshes in the volume data file.
  // This can be simplified by using ElementId and Mesh in the VolumeData class.
  std::vector<ElementId<Dim>> element_ids{};
  std::unordered_map<ElementId<Dim>, Mesh<Dim>> meshes{};
  element_ids.reserve(grid_names.size());
  for (const auto& grid_name : grid_names) {
    const ElementId<Dim> element_id(grid_name);
    element_ids.push_back(element_id);
    meshes[element_id] = h5::mesh_for_grid<Dim>(
        grid_name, grid_names, all_extents, all_bases, all_quadratures);
  }
  // Map the target points to element-logical coordinates. This selects the
  // subset of target points that are in the volume data file's elements.
  const auto element_logical_coords =
      element_logical_coordinates(element_ids, block_logical_coords);
  DataVector interpolated_data{};
  for (const auto& [element_id, point] : element_logical_coords) {
    const auto [offset, length] = h5::offset_and_length_for_grid(
        get_output(element_id), grid_names, all_extents);
    // Interpolate!
    // Possible optimization: rather than interpolating each tensor component
    // separately, we could interpolate all components at once. This would need
    // an offset and stride to be passed to the interpolator, since the tensor
    // components for all elements are stored contiguously.
    const intrp::Irregular<Dim> interpolant(meshes[element_id],
                                            point.element_logical_coords);
    const size_t num_element_target_points =
        point.element_logical_coords.begin()->size();
    if (interpolated_data.size() < num_element_target_points) {
      interpolated_data.destructive_resize(num_element_target_points);
    }
    for (size_t i = 0; i < tensor_components.size(); ++i) {
      auto output_data =
          gsl::make_span(interpolated_data.data(), num_element_target_points);
      const auto input_data =
          gsl::make_span(tensor_data[i].data() + offset, length);
      interpolant.interpolate(make_not_null(&output_data), input_data);
      for (size_t j = 0; j < num_element_target_points; ++j) {
        (*result)[i][point.offsets[j]] = interpolated_data[j];
        (*filled_data)[point.offsets[j]] = true;
      }
    }
  }
}
}  // namespace

template <size_t Dim>
std::vector<std::vector<double>> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    std::string subfile_name, int observation_step,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    const std::optional<size_t> num_threads) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  // Get the list of volume data files
  const std::vector<std::string> filenames =
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

  // Normalize subfile name
  if (subfile_name.size() > 4 &&
      subfile_name.substr(subfile_name.size() - 4) == ".vol") {
    subfile_name = subfile_name.substr(0, subfile_name.size() - 4);
  }
  if (subfile_name.front() != '/') {
    subfile_name = '/' + subfile_name;
  }

  // Retrieve info from the first volume file
  const h5::H5File<h5::AccessType::ReadOnly> first_h5file(filenames.front());
  const auto& first_volfile = first_h5file.get<h5::VolumeData>(subfile_name);
  const auto dim = first_volfile.get_dimension();
  if (dim != Dim) {
    ERROR_NO_TRACE("Mismatched dimensions: expected "
                   << Dim << "D volume data, but got " << dim << "D.");
  }
  // Get observation ID. This currently assumes that all volume files contain
  // the same observations. For generalizing to volume files across multiple
  // segments, see the Python function `Visualization.ReadH5:select_observation`
  // and possibly move it to C++.
  const auto obs_ids = first_volfile.list_observation_ids();
  if (observation_step < 0) {
    observation_step += static_cast<int>(obs_ids.size());
  }
  if (observation_step < 0 or
      static_cast<size_t>(observation_step) >= obs_ids.size()) {
    ERROR_NO_TRACE("Invalid observation step: "
                   << observation_step << ". There are " << obs_ids.size()
                   << " observations in the file.");
  }
  const size_t obs_id = obs_ids[static_cast<size_t>(observation_step)];
  // Get domain, time, functions of time
  const auto domain =
      deserialize<Domain<Dim>>(first_volfile.get_domain(obs_id)->data());
  const auto [time, functions_of_time] = [&first_volfile, &obs_id, &domain]() {
    if (domain.is_time_dependent()) {
      return std::make_pair(
          first_volfile.get_observation_value(obs_id),
          deserialize<domain::FunctionsOfTimeMap>(
              first_volfile.get_functions_of_time(obs_id)->data()));
    } else {
      return std::make_pair(0., domain::FunctionsOfTimeMap{});
    }
  }();
  first_h5file.close();

  // Look up block logical coordinates for all target points by mapping them
  // through the domain
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{};
  const size_t num_target_points = target_points[0].size();
  for (size_t d = 0; d < Dim; ++d) {
    const auto& target_coord = gsl::at(target_points, d);
    if (target_coord.size() != num_target_points) {
      ERROR_NO_TRACE("Mismatched number of target points: coordinate 0 has "
                     << num_target_points << " points, but coordinate " << d
                     << " has " << target_coord.size() << " points.");
    }
    inertial_coords.get(d).set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(target_coord.data()), num_target_points);
  }
  const auto block_logical_coords = block_logical_coordinates(
      domain, inertial_coords, time, functions_of_time);

  // Allocate memory for result
  std::vector<std::vector<double>> result{};
  result.reserve(tensor_components.size());
  for (size_t i = 0; i < tensor_components.size(); ++i) {
    result.emplace_back(num_target_points,
                        std::numeric_limits<double>::signaling_NaN());
  }
  std::vector<bool> filled_data(num_target_points, false);

  // Process all volume files. Parallelized with OpenMP if available.
  // Note: `break` is not allowed in OpenMP loops, so we use a shared bool to
  // skip the remaining iterations.
  bool all_data_filled = false;
#ifdef _OPENMP
  const size_t resolved_num_threads =
      num_threads.value_or(omp_get_max_threads());
#else
  if (num_threads.has_value()) {
    ERROR_NO_TRACE(
        "OpenMP is not available, so num_threads cannot be specified.");
  }
#endif  // _OPENMP
#pragma omp parallel for num_threads(resolved_num_threads) \
    shared(all_data_filled)
  for (const auto& filename : filenames) {
    if (all_data_filled) {
      continue;
    }
    interpolate_to_points(make_not_null(&result), make_not_null(&filled_data),
                          filename, subfile_name, obs_id, tensor_components,
                          block_logical_coords);
    // Terminate early if all data has been filled
    if (std::all_of(filled_data.begin(), filled_data.end(),
                    [](const bool filled) { return filled; })) {
      all_data_filled = true;
    }
  }
  return result;
}

// Generate instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::vector<std::vector<double>> interpolate_to_points<DIM(data)>( \
      const std::variant<std::vector<std::string>, std::string>&              \
          volume_files_or_glob,                                               \
      std::string subfile_name, int observation_step,                         \
      const std::vector<std::string>& tensor_components,                      \
      const std::array<std::vector<double>, DIM(data)>& target_points,        \
      const std::optional<size_t> num_threads);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace spectre::Exporter

#pragma GCC diagnostic pop
