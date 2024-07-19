// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/Exporter.hpp"

#include <csignal>  // For Blaze error handling without PCH
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "DataStructures/Tensor/EagerMath/CartesianToSpherical.hpp"
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
#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"
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
    const std::vector<BlockLogicalCoords<Dim>>& block_logical_coords,
    [[maybe_unused]] const size_t num_threads) {
  const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
  const auto& volfile = h5file.get<h5::VolumeData>(subfile_name);
  const auto grid_names = volfile.get_grid_names(obs_id);
  const auto all_extents = volfile.get_extents(obs_id);
  const auto all_bases = volfile.get_bases(obs_id);
  const auto all_quadratures = volfile.get_quadratures(obs_id);
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
  if (element_logical_coords.empty()) {
    return;
  }
  // Load the tensor data for all grids in the file because it's stored
  // contiguously
  std::vector<DataVector> tensor_data{};
  tensor_data.reserve(tensor_components.size());
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
      tensor_data.push_back(std::move(double_component_data));
    }
  }
  h5file.close();
#pragma omp parallel num_threads(num_threads)
  {
    DataVector interpolated_data{};
#pragma omp for
    for (const auto& element_id : element_ids) {
      const auto found_points = element_logical_coords.find(element_id);
      if (found_points == element_logical_coords.end()) {
        continue;
      }
      const auto& points = found_points->second;
      const auto [offset, length] = h5::offset_and_length_for_grid(
          get_output(element_id), grid_names, all_extents);
      // Interpolate!
      // Possible optimization: rather than interpolating each tensor component
      // separately, we could interpolate all components at once. This would
      // need an offset and stride to be passed to the interpolator, since the
      // tensor components for all elements are stored contiguously.
      const intrp::Irregular<Dim> interpolant(meshes[element_id],
                                              points.element_logical_coords);
      const size_t num_element_target_points =
          points.element_logical_coords.begin()->size();
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
          (*result)[i][points.offsets[j]] = interpolated_data[j];
          (*filled_data)[points.offsets[j]] = true;
        }
      }
    }  // omp for
  }  // omp parallel
}

// Data structure for extrapolation of tensor components into excisions
template <size_t NumExtrapolationAnchors>
struct ExtrapolationInfo {
  // Index of the target point in the result array (y target)
  size_t target_index;
  // Index of the first anchor point in the source array (y data)
  size_t source_index;
  // Coordinate of the target point (x target)
  double target_point;
  // Coordinates of the anchor points (x data)
  std::array<double, NumExtrapolationAnchors> anchors;
};

// Construct anchor points for extrapolation into excisions.
//
// Anchor points are placed radially in the grid frame around the excision
// sphere, which is spherical in the grid frame. Then the anchor points are
// added to the `block_logical_coords` so data is interpolated to them. Also an
// entry is added to `extrapolation_info` so the interpolated data on the anchor
// points can be extrapolated to the target point in the excision.
//
// This function returns `true` if the target point is inside the excision and
// the anchor points were added. Otherwise, returns `false` and the next
// excision should be tried.
//
// Alternatives and notes:
// - We could extrapolate directly from the nearest element, using the Lagrange
//   polynomial basis that the element already has. However, this is unstable
//   when the element is too small (possibly h-refined) and/or has log spacing.
//   In those cases the logical coordinate that we're extrapolating to can be
//   many logical element sizes away. Also, this relies on inverting the
//   neighboring block's map outside the block, which is not guaranteed to work.
// - We could construct anchor points in different frames, e.g. in the inertial
//   frame. These choices probably don't make a big difference.
// - We could do more performance optimizations for the case where the excision
//   sphere is spherical. E.g. in that case the radial anchor points for all
//   points are the same.
template <size_t Dim, size_t NumExtrapolationAnchors>
bool add_extrapolation_anchors(
    const gsl::not_null<std::vector<BlockLogicalCoords<Dim>>*>
        block_logical_coords,
    const gsl::not_null<
        std::vector<ExtrapolationInfo<NumExtrapolationAnchors>>*>
        extrapolation_info,
    const ExcisionSphere<Dim>& excision_sphere, const Domain<Dim>& domain,
    const tnsr::I<double, Dim, Frame::Inertial>& target_point,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const double extrapolation_spacing) {
  // Get spherical coordinates around the excision in distorted frame.
  // Note that the excision sphere doesn't have a distorted map, but its
  // grid-to-inertial map is just the neighboring block's
  // distorted-to-inertial map, so we can use either of those.
  auto x_distorted =
      [&excision_sphere, &target_point, &time,
       &functions_of_time]() -> tnsr::I<double, Dim, Frame::Distorted> {
    if (excision_sphere.is_time_dependent()) {
      const auto x_grid =
          excision_sphere.moving_mesh_grid_to_inertial_map().inverse(
              target_point, time, functions_of_time);
      ASSERT(x_grid.has_value(),
             "Failed to invert grid-to-inertial map of excision sphere at "
                 << excision_sphere.center() << " for point " << target_point
                 << ".");
      tnsr::I<double, Dim, Frame::Distorted> result{};
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d) = x_grid->get(d);
      }
      return result;
    } else {
      tnsr::I<double, Dim, Frame::Distorted> result{};
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d) = target_point.get(d);
      }
      return result;
    }
  }();
  for (size_t d = 0; d < Dim; ++d) {
    x_distorted.get(d) -= excision_sphere.center().get(d);
  }
  auto x_spherical_distorted = cartesian_to_spherical(x_distorted);
  // Construct anchor points in grid frame
  tnsr::I<DataVector, Dim, Frame::Spherical<Frame::Grid>>
      anchors_grid_spherical{NumExtrapolationAnchors};
  for (size_t i = 0; i < NumExtrapolationAnchors; ++i) {
    get<0>(anchors_grid_spherical)[i] =
        1. + static_cast<double>(i) * extrapolation_spacing;
  }
  get<0>(anchors_grid_spherical) *= excision_sphere.radius();
  // The grid-distorted map preserves angles
  if constexpr (Dim > 1) {
    get<1>(anchors_grid_spherical) = get<1>(x_spherical_distorted);
  }
  if constexpr (Dim > 2) {
    get<2>(anchors_grid_spherical) = get<2>(x_spherical_distorted);
  }
  auto anchors_grid = spherical_to_cartesian(anchors_grid_spherical);
  for (size_t d = 0; d < Dim; ++d) {
    anchors_grid.get(d) += excision_sphere.center().get(d);
  }
  // Map anchor points to block logical coordinates. These are needed for
  // interpolation.
  auto anchors_block_logical =
      block_logical_coordinates(domain, anchors_grid, time, functions_of_time);
  const size_t block_id = anchors_block_logical[0]->id.get_index();
  const auto& block = domain.blocks()[block_id];
  // Map anchor points to the distorted frame. We will extrapolate in
  // the distorted frame because we have the target point in the distorted
  // frame. The target point in the grid frame is undefined because there's
  // no grid-distorted map in the excision sphere. We could extrapolate in
  // the inertial frame, but that's just an unnecessary transformation.
  auto anchors_distorted =
      [&block, &anchors_grid, &time,
       &functions_of_time]() -> tnsr::I<DataVector, Dim, Frame::Distorted> {
    if (block.has_distorted_frame()) {
      return block.moving_mesh_grid_to_distorted_map()(anchors_grid, time,
                                                       functions_of_time);
    } else {
      tnsr::I<DataVector, Dim, Frame::Distorted> result{};
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d) = anchors_grid.get(d);
      }
      return result;
    }
  }();
  for (size_t d = 0; d < Dim; ++d) {
    anchors_distorted.get(d) -= excision_sphere.center().get(d);
  }
  // Compute magnitude(anchors_distorted) and store in std::array
  std::array<double, NumExtrapolationAnchors> radial_anchors_distorted_frame{};
  for (size_t i = 0; i < NumExtrapolationAnchors; ++i) {
    auto& radial_anchor_distorted_frame =
        gsl::at(radial_anchors_distorted_frame, i);
    radial_anchor_distorted_frame = square(get<0>(anchors_distorted)[i]);
    if constexpr (Dim > 1) {
      radial_anchor_distorted_frame += square(get<1>(anchors_distorted)[i]);
    }
    if constexpr (Dim > 2) {
      radial_anchor_distorted_frame += square(get<2>(anchors_distorted)[i]);
    }
    radial_anchor_distorted_frame = sqrt(radial_anchor_distorted_frame);
  }
  // Return false if the target point is not inside the excision. It would be
  // nice to do this earlier to avoid unnecessary work. Here we use the first
  // anchor point in the distorted frame, which is at the excision boundary in
  // the angular direction of the target point.
  const double excision_radius_distorted = radial_anchors_distorted_frame[0];
  if (get<0>(x_spherical_distorted) > excision_radius_distorted) {
    return false;
  }
  // Add the anchor points to the the interpolation target points
  for (size_t i = 0; i < NumExtrapolationAnchors; ++i) {
    ASSERT(anchors_block_logical[i].has_value(),
           "Extrapolation anchor point is not in any block. This should "
           "not happen.");
    block_logical_coords->push_back(std::move(anchors_block_logical[i]));
  }
  // Add the anchor points to the extrapolation info
  extrapolation_info->push_back(
      {// Target index is the point we're extrapolating to. It will be set
       // outside this function.
       std::numeric_limits<size_t>::max(),
       // Source index will be adjusted outside this function as well.
       extrapolation_info->size() * NumExtrapolationAnchors,
       // Target point and anchors are the radii in the distorted frame
       get<0>(x_spherical_distorted),
       std::move(radial_anchors_distorted_frame)});
  return true;
}

// Determines the selected observation ID in the volume data file, given either
// an `ObservationId` directly or an `ObservationStep`.
struct SelectObservation {
  size_t operator()(const ObservationId observation_id) const {
    return observation_id.value;
  }
  size_t operator()(const ObservationStep observation_step) const {
    int step = observation_step.value;
    const auto obs_ids = volfile.list_observation_ids();
    if (step < 0) {
      step += static_cast<int>(obs_ids.size());
    }
    if (step < 0 or static_cast<size_t>(step) >= obs_ids.size()) {
      ERROR_NO_TRACE("Invalid observation step: "
                     << observation_step.value << ". There are "
                     << obs_ids.size() << " observations in the file.");
    }
    return obs_ids[static_cast<size_t>(step)];
  }
  const h5::VolumeData& volfile;
};

}  // namespace

template <size_t Dim>
std::vector<std::vector<double>> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name,
    const std::variant<ObservationId, ObservationStep>& observation,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    const bool extrapolate_into_excisions,
    const std::optional<size_t> num_threads) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  // Resolve number of threads to use in OpenMP parallelization
#ifdef _OPENMP
  const size_t resolved_num_threads =
      num_threads.value_or(omp_get_max_threads());
#else
  if (num_threads.has_value()) {
    ERROR_NO_TRACE(
        "OpenMP is not available, so num_threads cannot be specified.");
  }
  const size_t resolved_num_threads = 1;
#endif  // _OPENMP

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

  // Retrieve info from the first volume file
  const h5::H5File<h5::AccessType::ReadOnly> first_h5file(filenames.front());
  const auto& first_volfile = first_h5file.get<h5::VolumeData>(subfile_name);
  const auto dim = first_volfile.get_dimension();
  if (dim != Dim) {
    ERROR_NO_TRACE("Mismatched dimensions: expected "
                   << Dim << "D volume data, but got " << dim << "D.");
  }
  // Get observation ID
  // This currently assumes that all volume files contain the same observations,
  // so we only look into the first file. For generalizing to volume files
  // across multiple segments, see the Python function
  // `Visualization.ReadH5:select_observation` and possibly move it to C++.
  const size_t obs_id =
      std::visit(SelectObservation{first_volfile}, observation);
  // Get domain, time, functions of time
  const auto domain =
      deserialize<Domain<Dim>>(first_volfile.get_domain(obs_id)->data());
  const auto time_and_fot = [&first_volfile, &obs_id, &domain]() {
    if (domain.is_time_dependent()) {
      return std::make_pair(
          first_volfile.get_observation_value(obs_id),
          deserialize<domain::FunctionsOfTimeMap>(
              first_volfile.get_functions_of_time(obs_id)->data()));
    } else {
      return std::make_pair(0., domain::FunctionsOfTimeMap{});
    }
  }();
  const double time = time_and_fot.first;
  const auto& functions_of_time = time_and_fot.second;
  first_h5file.close();

  // Check target points have the same number of points in each dimension
  const size_t num_target_points = target_points[0].size();
  for (size_t d = 0; d < Dim; ++d) {
    const auto& target_coord = gsl::at(target_points, d);
    if (target_coord.size() != num_target_points) {
      ERROR_NO_TRACE("Mismatched number of target points: coordinate 0 has "
                     << num_target_points << " points, but coordinate " << d
                     << " has " << target_coord.size() << " points.");
    }
  }

  // Look up block logical coordinates for all target points by mapping them
  // through the domain. This is the most expensive part of the function, so we
  // parallelize the loop.
  std::vector<BlockLogicalCoords<Dim>> block_logical_coords(num_target_points);
  // We also set up the extrapolation into excisions here. Anchor points are
  // added to the `block_logical_coords` and additional information is collected
  // in `extrapolation_info` for later extrapolation.
  constexpr size_t num_extrapolation_anchors = 8;
  const double extrapolation_spacing = 0.3;
  std::vector<ExtrapolationInfo<num_extrapolation_anchors>>
      extrapolation_info{};
#pragma omp parallel num_threads(resolved_num_threads)
  {
    // Set up thread-local variables
    tnsr::I<double, Dim, Frame::Inertial> target_point{};
    std::vector<BlockLogicalCoords<Dim>> extra_block_logical_coords{};
    std::vector<ExtrapolationInfo<num_extrapolation_anchors>>
        extra_extrapolation_info{};
#pragma omp for nowait
    for (size_t s = 0; s < num_target_points; ++s) {
      for (size_t d = 0; d < Dim; ++d) {
        target_point.get(d) = gsl::at(target_points, d)[s];
      }
      for (const auto& block : domain.blocks()) {
        auto x_logical = block_logical_coordinates_single_point(
            target_point, block, time, functions_of_time);
        if (x_logical.has_value()) {
          block_logical_coords[s] = {domain::BlockId(block.id()),
                                     std::move(x_logical.value())};
          break;
        }
      }  // for blocks
      if (block_logical_coords[s].has_value() or
          not extrapolate_into_excisions) {
        continue;
      }
      // The point wasn't found in any block. Check if it's in an excision and
      // set up extrapolation if requested.
      for (const auto& [name, excision_sphere] : domain.excision_spheres()) {
        if (add_extrapolation_anchors(
                make_not_null(&extra_block_logical_coords),
                make_not_null(&extra_extrapolation_info), excision_sphere,
                domain, target_point, time, functions_of_time,
                extrapolation_spacing)) {
          extra_extrapolation_info.back().target_index = s;
          break;
        }
      }
    }  // omp for target points
#pragma omp critical
    {
      // Append the extra block logical coordinates and extrapolation info from
      // this thread to the global vectors. Also set the source index to the
      // index in `block_logical_coords` where we're going to insert the new
      // coordinates.
      for (auto& info : extra_extrapolation_info) {
        info.source_index += block_logical_coords.size();
      }
      block_logical_coords.insert(block_logical_coords.end(),
                                  extra_block_logical_coords.begin(),
                                  extra_block_logical_coords.end());
      extrapolation_info.insert(extrapolation_info.end(),
                                extra_extrapolation_info.begin(),
                                extra_extrapolation_info.end());
    }  // omp critical
  }  // omp parallel

  // Allocate memory for result
  std::vector<std::vector<double>> result{};
  result.reserve(tensor_components.size());
  for (size_t i = 0; i < tensor_components.size(); ++i) {
    result.emplace_back(block_logical_coords.size(),
                        std::numeric_limits<double>::signaling_NaN());
  }
  std::vector<bool> filled_data(block_logical_coords.size(), false);

  // Process all volume files in serial, because loading data with H5 must be
  // done in serial anyway. Instead, the loop over elements within each file is
  // parallelized with OpenMP.
  for (const auto& filename : filenames) {
    interpolate_to_points(make_not_null(&result), make_not_null(&filled_data),
                          filename, subfile_name, obs_id, tensor_components,
                          block_logical_coords, resolved_num_threads);
    // Terminate early if all data has been filled
    if (std::all_of(filled_data.begin(), filled_data.end(),
                    [](const bool filled) { return filled; })) {
      break;
    }
  }

  if (extrapolate_into_excisions) {
    // Extrapolate into excisions from the anchor points
#pragma omp parallel for num_threads(resolved_num_threads)
    for (const auto& extrapolation : extrapolation_info) {
      double extrapolation_error = 0.;
      for (size_t i = 0; i < tensor_components.size(); ++i) {
        intrp::polynomial_interpolation<num_extrapolation_anchors - 1>(
            make_not_null(&result[i][extrapolation.target_index]),
            make_not_null(&extrapolation_error), extrapolation.target_point,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            gsl::make_span(result[i].data() + extrapolation.source_index,
                           num_extrapolation_anchors),
            gsl::make_span(extrapolation.anchors.data(),
                           num_extrapolation_anchors));
      }
    }
    // Clear the anchor points from the result
    for (size_t i = 0; i < tensor_components.size(); ++i) {
      result[i].resize(num_target_points);
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
      const std::string& subfile_name,                                        \
      const std::variant<ObservationId, ObservationStep>& observation,        \
      const std::vector<std::string>& tensor_components,                      \
      const std::array<std::vector<double>, DIM(data)>& target_points,        \
      bool extrapolate_into_excisions,                                        \
      const std::optional<size_t> num_threads);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace spectre::Exporter

#pragma GCC diagnostic pop
