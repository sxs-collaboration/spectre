// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/PointwiseInterpolator.hpp"

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
#include "IO/Exporter/SelectObservation.hpp"
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

size_t resolve_num_threads(const std::optional<size_t> num_threads) {
  // Resolve number of threads to use in OpenMP parallelization
#ifdef _OPENMP
  return num_threads.value_or(omp_get_max_threads());
#else
  if (num_threads.has_value()) {
    ERROR_NO_TRACE(
        "OpenMP is not available, so num_threads cannot be specified.");
  }
  return 1;
#endif  // _OPENMP
}

template <size_t Dim>
auto load_grids(const h5::VolumeData& volfile, const size_t obs_id) {
  const auto grid_names = volfile.get_grid_names(obs_id);
  const auto all_extents = volfile.get_extents(obs_id);
  const auto all_bases = volfile.get_bases(obs_id);
  const auto all_quadratures = volfile.get_quadratures(obs_id);
  // Reconstruct element IDs & meshes in the volume data file.
  // This can be simplified by using ElementId and Mesh in the VolumeData class.
  std::vector<ElementId<Dim>> element_ids{};
  std::unordered_map<ElementId<Dim>, std::tuple<Mesh<Dim>, size_t, size_t>>
      meshes{};
  element_ids.reserve(grid_names.size());
  for (const auto& grid_name : grid_names) {
    const ElementId<Dim> element_id(grid_name);
    element_ids.push_back(element_id);
    get<0>(meshes[element_id]) = h5::mesh_for_grid<Dim>(
        grid_name, grid_names, all_extents, all_bases, all_quadratures);
    const auto [offset, length] = h5::offset_and_length_for_grid(
        get_output(element_id), grid_names, all_extents);
    get<1>(meshes[element_id]) = offset;
    get<2>(meshes[element_id]) = length;
  }
  return std::make_pair(std::move(element_ids), std::move(meshes));
}

std::vector<std::variant<DataVector, std::vector<float>>> load_tensor_data(
    const h5::VolumeData& volfile, const size_t obs_id,
    const std::vector<std::string>& tensor_components) {
  std::vector<std::variant<DataVector, std::vector<float>>> tensor_data{};
  tensor_data.reserve(tensor_components.size());
  for (const auto& tensor_component : tensor_components) {
    tensor_data.push_back(
        volfile.get_tensor_component(obs_id, tensor_component).data);
  }
  return tensor_data;
}

template <typename ResultDataType, size_t Dim>
void interpolate_to_points_impl(
    const gsl::not_null<std::vector<ResultDataType>*> result,
    const gsl::not_null<std::vector<bool>*> filled_data,
    const std::unordered_map<ElementId<Dim>, ElementLogicalCoordHolder<Dim>>&
        element_logical_coords,
    const std::vector<ElementId<Dim>>& element_ids,
    const std::unordered_map<ElementId<Dim>,
                             std::tuple<Mesh<Dim>, size_t, size_t>>& meshes,
    const std::vector<std::variant<DataVector, std::vector<float>>>&
        tensor_data,
    [[maybe_unused]] const size_t resolved_num_threads) {
#pragma omp parallel num_threads(resolved_num_threads)
  {
    DataVector interpolated_data_double{};
    std::vector<float> interpolated_data_float{};
#pragma omp for
    for (const auto& element_id : element_ids) {
      const auto found_points = element_logical_coords.find(element_id);
      if (found_points == element_logical_coords.end()) {
        continue;
      }
      const auto& points = found_points->second;
      const auto& [mesh, offset, length] = meshes.at(element_id);
      // Interpolate!
      // Possible optimization: rather than interpolating each tensor component
      // separately, we could interpolate all components at once. This would
      // need an offset and stride to be passed to the interpolator, since the
      // tensor components for all elements are stored contiguously.
      const intrp::Irregular<Dim> interpolant(mesh,
                                              points.element_logical_coords);
      const size_t num_element_target_points =
          points.element_logical_coords.begin()->size();
      for (size_t i = 0; i < tensor_data.size(); ++i) {
        const auto& component = tensor_data[i];
        if (std::holds_alternative<DataVector>(component)) {
          if (interpolated_data_double.size() < num_element_target_points) {
            interpolated_data_double.destructive_resize(
                num_element_target_points);
          }
          auto output_data = gsl::make_span(interpolated_data_double.data(),
                                            num_element_target_points);
          const auto input_data = gsl::make_span(
              std::get<DataVector>(component).data() + offset, length);
          interpolant.interpolate(make_not_null(&output_data), input_data);
          for (size_t j = 0; j < num_element_target_points; ++j) {
            (*result)[i][points.offsets[j]] = interpolated_data_double[j];
            (*filled_data)[points.offsets[j]] = true;
          }
        } else {
          interpolated_data_float.resize(num_element_target_points);
          auto output_data = gsl::make_span(interpolated_data_float.data(),
                                            num_element_target_points);
          const auto input_data = gsl::make_span(
              // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
              std::get<std::vector<float>>(component).data() + offset, length);
          interpolant.interpolate(make_not_null(&output_data), input_data);
          for (size_t j = 0; j < num_element_target_points; ++j) {
            (*result)[i][points.offsets[j]] = interpolated_data_float[j];
            (*filled_data)[points.offsets[j]] = true;
          }
        }
      }
    }  // omp for
  }  // omp parallel
}

template <size_t Dim>
void interpolate_to_point_impl(
    const gsl::not_null<std::vector<double>*> result,
    const tnsr::I<double, Dim, Frame::ElementLogical>& x_element_logical,
    const Mesh<Dim>& mesh, const size_t offset, const size_t length,
    const std::vector<std::variant<DataVector, std::vector<float>>>&
        tensor_data) {
  const intrp::Irregular<Dim> interpolant(mesh, x_element_logical);
  const size_t num_components = tensor_data.size();
  result->resize(num_components);
  for (size_t i = 0; i < num_components; ++i) {
    const auto& component = tensor_data[i];
    if (std::holds_alternative<DataVector>(component)) {
      const auto input_data = gsl::make_span(
          std::get<DataVector>(component).data() + offset, length);
      auto output_data = gsl::make_span(&(*result)[i], 1);
      interpolant.interpolate(make_not_null(&output_data), input_data);
    } else {
      const auto input_data = gsl::make_span(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          std::get<std::vector<float>>(component).data() + offset, length);
      float output_value = std::numeric_limits<float>::signaling_NaN();
      auto output_data = gsl::make_span(&output_value, 1);
      interpolant.interpolate(make_not_null(&output_data), input_data);
      (*result)[i] = output_value;
    }
  }
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

template <typename ResultDataType, size_t NumExtrapolationAnchors>
void extrapolate_into_excisions_impl(
    const gsl::not_null<std::vector<ResultDataType>*> result,
    const std::vector<ExtrapolationInfo<NumExtrapolationAnchors>>&
        extrapolation_info,
    [[maybe_unused]] const size_t resolved_num_threads) {
#pragma omp parallel for num_threads(resolved_num_threads)
  for (const auto& extrapolation : extrapolation_info) {
    double extrapolation_error = 0.;
    for (size_t i = 0; i < result->size(); ++i) {
      intrp::polynomial_interpolation<NumExtrapolationAnchors - 1>(
          make_not_null(&(*result)[i][extrapolation.target_index]),
          make_not_null(&extrapolation_error), extrapolation.target_point,
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          gsl::make_span((*result)[i].data() + extrapolation.source_index,
                         NumExtrapolationAnchors),
          gsl::make_span(extrapolation.anchors.data(),
                         NumExtrapolationAnchors));
    }
  }  // omp for extrapolation_info
}

template <size_t Dim>
auto parse_volume_files(const std::variant<std::vector<std::string>,
                                           std::string>& volume_files_or_glob,
                        const std::string& subfile_name,
                        const ObservationVariant& observation) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  // Get the list of volume data files
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
  auto domain =
      deserialize<Domain<Dim>>(first_volfile.get_domain(obs_id)->data());
  auto time_and_fot = [&first_volfile, &obs_id, &domain]() {
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
  return std::make_tuple(std::move(filenames), obs_id, time_and_fot.first,
                         std::move(domain), std::move(time_and_fot.second));
}

template <typename DataType, size_t Dim, typename Frame>
auto block_logical_coordinates(
    const tnsr::I<DataType, Dim, Frame>& target_points,
    const Domain<Dim>& domain, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const bool extrapolate_into_excisions, const bool error_on_missing_points,
    [[maybe_unused]] const size_t resolved_num_threads) {
  const size_t num_target_points = get_size(get<0>(target_points));
  // Look up block logical coordinates for all target points by mapping them
  // through the domain. This is the most expensive part of the function, so we
  // parallelize the loop.
  std::vector<BlockLogicalCoords<Dim>> block_logical_coords(num_target_points);
  // We also set up the extrapolation into excisions here. Anchor points are
  // added to the `block_logical_coords` and additional information is collected
  // in `extrapolation_info` for later extrapolation.
  constexpr size_t num_extrapolation_anchors = 8;
  [[maybe_unused]] const double extrapolation_spacing = 0.3;
  std::vector<ExtrapolationInfo<num_extrapolation_anchors>>
      extrapolation_info{};
  if constexpr (not std::is_same_v<Frame, ::Frame::Inertial>) {
    if (extrapolate_into_excisions) {
      ERROR(
          "Extrapolation into excisions is only supported in the inertial "
          "frame at the moment.");
    }
  }
#pragma omp parallel num_threads(resolved_num_threads)
  {
    // Set up thread-local variables
    tnsr::I<double, Dim, Frame> target_point{};
    std::vector<BlockLogicalCoords<Dim>> extra_block_logical_coords{};
    std::vector<ExtrapolationInfo<num_extrapolation_anchors>>
        extra_extrapolation_info{};
    // Keep track of a priority order for blocks to accelerate the search.
    // This helps when the target points are ordered so that they are likely to
    // be in the same block as the previous point.
    std::vector<size_t> block_order(domain.blocks().size());
    std::iota(block_order.begin(), block_order.end(), 0);
#pragma omp for
    for (size_t s = 0; s < num_target_points; ++s) {
      for (size_t d = 0; d < Dim; ++d) {
        target_point.get(d) = get_element(target_points.get(d), s);
      }
      block_logical_coords[s] = block_logical_coordinates_single_point(
          target_point, domain, time, functions_of_time,
          make_not_null(&block_order));
      if (block_logical_coords[s].has_value()) {
        continue;
      }
      if (not extrapolate_into_excisions) {
        // The point wasn't found in any block and we're not extrapolating.
        // Check if we should throw an error or just skip this point.
        if (error_on_missing_points) {
          ERROR("Point is not in any block:\n" << target_point);
        } else {
          continue;
        }
      }
      // The point wasn't found in any block. Check if it's in an excision and
      // set up extrapolation if requested.
      if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
        bool found_in_excision = false;
        for (const auto& [name, excision_sphere] : domain.excision_spheres()) {
          if (add_extrapolation_anchors(
                  make_not_null(&extra_block_logical_coords),
                  make_not_null(&extra_extrapolation_info), excision_sphere,
                  domain, target_point, time, functions_of_time,
                  extrapolation_spacing)) {
            extra_extrapolation_info.back().target_index = s;
            found_in_excision = true;
            break;
          }
        }
        if (not found_in_excision and error_on_missing_points) {
          ERROR("Point is not in any block or excision:\n" << target_point);
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
  return std::make_tuple(std::move(block_logical_coords),
                         std::move(extrapolation_info));
}

}  // namespace

template <typename ResultDataType, size_t Dim, typename Frame>
void interpolate_to_points(
    const gsl::not_null<std::vector<ResultDataType>*> result,
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const tnsr::I<DataVector, Dim, Frame>& target_points,
    const bool extrapolate_into_excisions, const bool error_on_missing_points,
    const std::optional<size_t> num_threads) {
  const size_t num_target_points = get<0>(target_points).size();
  const auto [filenames, obs_id, time, domain, functions_of_time] =
      parse_volume_files<Dim>(volume_files_or_glob, subfile_name, observation);

  const size_t resolved_num_threads = resolve_num_threads(num_threads);

  const auto [block_logical_coords, extrapolation_info] =
      block_logical_coordinates(target_points, domain, time, functions_of_time,
                                extrapolate_into_excisions,
                                error_on_missing_points, resolved_num_threads);
  const size_t num_interpolation_points = block_logical_coords.size();

  // Allocate memory for result
  const size_t num_components = tensor_components.size();
  result->resize(num_components);
  for (size_t i = 0; i < num_components; ++i) {
    ResultDataType& component = (*result)[i];
    if constexpr (std::is_same_v<ResultDataType, DataVector>) {
      component.destructive_resize(num_interpolation_points);
    } else {
      component.resize(num_interpolation_points);
    }
    std::fill(component.begin(), component.end(),
              std::numeric_limits<double>::signaling_NaN());
  }
  std::vector<bool> filled_data(num_interpolation_points, false);

  // Process all volume files in serial, because loading data with H5 must be
  // done in serial anyway. Instead, the loop over elements within each file is
  // parallelized with OpenMP.
  for (const auto& filename : filenames) {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
    const auto& volfile = h5file.get<h5::VolumeData>(subfile_name);
    const auto [element_ids, meshes] = load_grids<Dim>(volfile, obs_id);

    // Map the target points to element-logical coordinates. This selects the
    // subset of target points that are in the volume data file's elements.
    const auto element_logical_coords =
        element_logical_coordinates(element_ids, block_logical_coords);
    if (element_logical_coords.empty()) {
      continue;
    }

    // Load the tensor data for all grids in the file because it's stored
    // contiguously
    const auto tensor_data =
        load_tensor_data(volfile, obs_id, tensor_components);
    h5file.close();

    // Interpolate the tensor data to the target points
    interpolate_to_points_impl(result, make_not_null(&filled_data),
                               element_logical_coords, element_ids, meshes,
                               tensor_data, resolved_num_threads);

    // Terminate early if all data has been filled
    if (std::all_of(filled_data.begin(), filled_data.end(),
                    [](const bool filled) { return filled; })) {
      break;
    }
  }

  if (extrapolate_into_excisions and not extrapolation_info.empty()) {
    extrapolate_into_excisions_impl(result, extrapolation_info,
                                    resolved_num_threads);
    // Clear the anchor points from the result
    for (size_t i = 0; i < result->size(); ++i) {
      ResultDataType& component = (*result)[i];
      if constexpr (std::is_same_v<ResultDataType, DataVector>) {
        // Can't use `destructive_resize` because we want to preserve the
        // leading elements of the DataVector
        DataVector new_component(num_target_points);
        std::copy(
            component.begin(),
            component.begin() + static_cast<std::ptrdiff_t>(num_target_points),
            new_component.begin());
        component = std::move(new_component);
      } else {
        component.resize(num_target_points);
      }
    }
  }
}

template <size_t Dim, typename Frame>
PointwiseInterpolator<Dim, Frame>::PointwiseInterpolator(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components) {
  auto bindings =
      parse_volume_files<Dim>(volume_files_or_glob, subfile_name, observation);
  const auto& filenames = std::get<0>(bindings);
  obs_id_ = std::get<1>(bindings);
  time_ = std::get<2>(bindings);
  domain_ = std::move(std::get<3>(bindings));
  functions_of_time_ = std::move(std::get<4>(bindings));

  // Load tensor data into memory
  element_ids_.reserve(filenames.size());
  element_search_trees_.reserve(filenames.size());
  meshes_.reserve(filenames.size());
  tensor_data_.reserve(filenames.size());
  for (const auto& filename : filenames) {
    const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
    const auto& volfile = h5file.get<h5::VolumeData>(subfile_name);
    const auto [element_ids, meshes] = load_grids<Dim>(volfile, obs_id_);
    element_ids_.push_back(std::move(element_ids));
    element_search_trees_.push_back(
        domain::index_element_ids(element_ids_.back()));
    meshes_.push_back(std::move(meshes));
    tensor_data_.push_back(
        load_tensor_data(volfile, obs_id_, tensor_components));
    h5file.close();
  }
}

template <size_t Dim, typename Frame>
void PointwiseInterpolator<Dim, Frame>::interpolate_to_points(
    const gsl::not_null<std::vector<DataVector>*> result,
    const tnsr::I<DataVector, Dim, Frame>& target_points,
    const bool extrapolate_into_excisions, const bool error_on_missing_points,
    const std::optional<size_t> num_threads) const {
  ASSERT(not tensor_data_.empty(),
         "PointwiseInterpolator has not been initialized with tensor data.");
  const size_t resolved_num_threads = resolve_num_threads(num_threads);
  const size_t num_target_points = get<0>(target_points).size();

  // Map the target points through the domain
  const auto [block_logical_coords, extrapolation_info] =
      block_logical_coordinates(target_points, domain_, time_,
                                functions_of_time_, extrapolate_into_excisions,
                                error_on_missing_points, resolved_num_threads);
  const size_t num_interpolation_points = block_logical_coords.size();

  // Allocate memory for result
  const size_t num_components = tensor_data_.front().size();
  result->resize(num_components);
  for (size_t i = 0; i < num_components; ++i) {
    DataVector& component = (*result)[i];
    component.destructive_resize(num_interpolation_points);
    std::fill(component.begin(), component.end(),
              std::numeric_limits<double>::signaling_NaN());
  }
  std::vector<bool> filled_data(num_interpolation_points, false);

  // Process all volume files in serial
  for (size_t file_id = 0; file_id < element_ids_.size(); ++file_id) {
    const auto& element_ids = element_ids_[file_id];
    const auto& meshes = meshes_[file_id];
    const auto& tensor_data = tensor_data_[file_id];

    // Map the target points to element-logical coordinates. This selects the
    // subset of target points that are in the volume data file's elements.
    const auto element_logical_coords =
        element_logical_coordinates(element_ids, block_logical_coords);
    if (element_logical_coords.empty()) {
      continue;
    }

    // Interpolate the tensor data to the target points
    interpolate_to_points_impl(result, make_not_null(&filled_data),
                               element_logical_coords, element_ids, meshes,
                               tensor_data, resolved_num_threads);

    // Terminate early if all data has been filled
    if (std::all_of(filled_data.begin(), filled_data.end(),
                    [](const bool filled) { return filled; })) {
      break;
    }
  }

  if (extrapolate_into_excisions) {
    extrapolate_into_excisions_impl(result, extrapolation_info,
                                    resolved_num_threads);
    // Clear the anchor points from the result
    for (size_t i = 0; i < result->size(); ++i) {
      DataVector& component = (*result)[i];
      component.destructive_resize(num_target_points);
    }
  }
}

template <size_t Dim, typename Frame>
void PointwiseInterpolator<Dim, Frame>::interpolate_to_point(
    const gsl::not_null<std::vector<double>*> result,
    const tnsr::I<double, Dim, Frame>& target_point,
    const std::optional<gsl::not_null<std::vector<size_t>*>> block_order)
    const {
  ASSERT(not tensor_data_.empty(),
         "PointwiseInterpolator has not been initialized with tensor data.");
  const auto block_logical_coords = block_logical_coordinates_single_point(
      target_point, domain_, time_, functions_of_time_, block_order);
  if (not block_logical_coords.has_value()) {
    ERROR("Point is not in any block:\n" << target_point);
  }
  interpolate_to_point(result, block_logical_coords.value());
}

template <size_t Dim, typename Frame>
void PointwiseInterpolator<Dim, Frame>::interpolate_to_point(
    const gsl::not_null<std::vector<double>*> result,
    const IdPair<domain::BlockId, tnsr::I<double, Dim, ::Frame::BlockLogical>>&
        target_point) const {
  ASSERT(not tensor_data_.empty(),
         "PointwiseInterpolator has not been initialized with tensor data.");
  for (size_t file_id = 0; file_id < element_ids_.size(); ++file_id) {
    const auto& element_search_tree = element_search_trees_[file_id];
    const auto& meshes = meshes_[file_id];
    const auto& tensor_data = tensor_data_[file_id];
    const auto element_logical_coords =
        element_logical_coordinates(target_point, element_search_tree);
    if (not element_logical_coords.has_value()) {
      continue;
    }
    const auto& element_id = element_logical_coords.value().first;
    const auto& x_element_logical = element_logical_coords.value().second;
    const auto& mesh_offset_length = meshes.at(element_id);
    interpolate_to_point_impl(
        result, x_element_logical, get<0>(mesh_offset_length),
        get<1>(mesh_offset_length), get<2>(mesh_offset_length), tensor_data);
    break;
  }
}

// Generate instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void interpolate_to_points<DTYPE(data), DIM(data), FRAME(data)>(   \
      gsl::not_null<std::vector<DTYPE(data)>*> result,                        \
      const std::variant<std::vector<std::string>, std::string>&              \
          volume_files_or_glob,                                               \
      const std::string& subfile_name, const ObservationVariant& observation, \
      const std::vector<std::string>& tensor_components,                      \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& target_points,       \
      bool extrapolate_into_excisions, bool error_on_missing_points,          \
      std::optional<size_t> num_threads);

#define INSTANTIATE_CLASSES(_, data) \
  template class PointwiseInterpolator<DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (DataVector, std::vector<double>))
GENERATE_INSTANTIATIONS(INSTANTIATE_CLASSES, (1, 2, 3),
                        (Frame::Grid, Frame::Inertial))

#undef INSTANTIATE
#undef INSTANTIATE_CLASSES
#undef DIM
#undef FRAME
#undef DTYPE

}  // namespace spectre::Exporter

#pragma GCC diagnostic pop
