// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoGridHelpers.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"  // IWYU pragma: keep
#include "Domain/Element.hpp"    // IWYU pragma: keep
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace SlopeLimiters {
namespace Weno_detail {

template <size_t VolumeDim>
bool check_element_has_one_similar_neighbor_in_direction(
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction) noexcept {
  const auto& neighbors = element.neighbors().at(direction);
  if (neighbors.size() > 1) {
    // More than one neighbor
    return false;
  } else {
    const auto& orientation_map = neighbors.orientation();
    const auto& neighbor_segment_ids = neighbors.ids().cbegin()->segment_ids();
    const auto reoriented_neighbor_segment_ids =
        orientation_map.inverse_map()(neighbor_segment_ids);
    for (size_t d = 0; d < VolumeDim; ++d) {
      if (gsl::at(reoriented_neighbor_segment_ids, d).refinement_level() !=
          gsl::at(element.id().segment_ids(), d).refinement_level()) {
        // One neighbor, but of a different refinement level
        return false;
      }
    }
  }
  return true;
}

template <size_t VolumeDim>
std::array<DataVector, VolumeDim> neighbor_grid_points_in_local_logical_coords(
    const Mesh<VolumeDim>& local_mesh, const Mesh<VolumeDim>& neighbor_mesh,
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction_to_neighbor) noexcept {
  ASSERT(check_element_has_one_similar_neighbor_in_direction(
             element, direction_to_neighbor),
         "Found some amount of h-refinement; this is not supported");

  constexpr double logical_element_width = 2.0;
  std::array<DataVector, VolumeDim> result{{{}}};
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto neighbor_mesh_1d = neighbor_mesh.slice_through(d);
    if (d == direction_to_neighbor.dimension()) {
      // Coordinates normal to the interface cannot line up, so we must provide
      // the coordinates to interpolate to
      gsl::at(result, d) =
          get<0>(logical_coordinates(neighbor_mesh_1d)) +
          (direction_to_neighbor.side() == Side::Upper ? 1.0 : -1.0) *
              logical_element_width;
    } else if (neighbor_mesh_1d != local_mesh.slice_through(d)) {
      // Coordinates parallel to the interface may or may not line up
      gsl::at(result, d) = get<0>(logical_coordinates(neighbor_mesh_1d));
    }
  }
  return result;
}

template <size_t VolumeDim>
std::array<DataVector, VolumeDim> local_grid_points_in_neighbor_logical_coords(
    const Mesh<VolumeDim>& local_mesh, const Mesh<VolumeDim>& neighbor_mesh,
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction_to_neighbor) noexcept {
  ASSERT(check_element_has_one_similar_neighbor_in_direction(
             element, direction_to_neighbor),
         "Found some amount of h-refinement; this is not supported");

  constexpr double logical_element_width = 2.0;
  std::array<DataVector, VolumeDim> result{{{}}};
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto local_mesh_1d = local_mesh.slice_through(d);
    if (d == direction_to_neighbor.dimension()) {
      // Coordinates normal to the interface cannot line up, so we must provide
      // the coordinates to interpolate to
      gsl::at(result, d) =
          get<0>(logical_coordinates(local_mesh_1d)) +
          (direction_to_neighbor.side() == Side::Upper ? -1.0 : 1.0) *
              logical_element_width;
    } else if (local_mesh_1d != neighbor_mesh.slice_through(d)) {
      // Coordinates parallel to the interface may or may not line up
      gsl::at(result, d) = get<0>(logical_coordinates(local_mesh_1d));
    }
  }
  return result;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template bool check_element_has_one_similar_neighbor_in_direction(    \
      const Element<DIM(data)>&, const Direction<DIM(data)>&) noexcept; \
  template std::array<DataVector, DIM(data)>                            \
  neighbor_grid_points_in_local_logical_coords(                         \
      const Mesh<DIM(data)>&, const Mesh<DIM(data)>&,                   \
      const Element<DIM(data)>&, const Direction<DIM(data)>&) noexcept; \
  template std::array<DataVector, DIM(data)>                            \
  local_grid_points_in_neighbor_logical_coords(                         \
      const Mesh<DIM(data)>&, const Mesh<DIM(data)>&,                   \
      const Element<DIM(data)>&, const Direction<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Weno_detail
}  // namespace SlopeLimiters
