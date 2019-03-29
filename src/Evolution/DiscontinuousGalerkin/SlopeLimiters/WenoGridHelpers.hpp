// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Direction;
template <size_t>
class Element;
template <size_t>
class Mesh;
/// \endcond

namespace SlopeLimiters {
namespace Weno_detail {

// Check that an element has just one neighbor in a particular direction, and
// that this neighbor has the same refinement level as the element.
template <size_t VolumeDim>
bool check_element_has_one_similar_neighbor_in_direction(
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction) noexcept;

// Compute the coordinate positions of a neighbor element's grid points in the
// logical coordinates of the local element.
//
// However, the results are organized to match the expected input of a
// RegularGridInterpolator constructor:
// - The coordinates are given as an array of 1D coordinate values.
// - For any dimension where the local and neighbor meshes share the same 1D
//   submesh, the array is empty. (This tells the RegularGridInterpolator that
//   no interpolation needs to be done in this dimension.)
//
// Limitations:
// - No support for h-refinement. It is ASSERT'ed that the element has only one
//   neighbor in the specified direction, and that this neighbor is of the same
//   refinement level as the local element.
// - Assumes a uniform rectilinear grid. No support for elements of different
//   sizes (as could occur if all elements have the same refinement level but
//   different blocks have different physical size), or curvilinear elements.
//   This is not checked.
template <size_t VolumeDim>
std::array<DataVector, VolumeDim> neighbor_grid_points_in_local_logical_coords(
    const Mesh<VolumeDim>& local_mesh, const Mesh<VolumeDim>& neighbor_mesh,
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction_to_neighbor) noexcept;

// Compute the coordinate positions of the local element's grid points in the
// logical coordinates of the a neighbor element.
//
// See documentation of `neighbor_grid_points_in_local_logical_coords` for
// further details and limitations.
template <size_t VolumeDim>
std::array<DataVector, VolumeDim> local_grid_points_in_neighbor_logical_coords(
    const Mesh<VolumeDim>& local_mesh, const Mesh<VolumeDim>& neighbor_mesh,
    const Element<VolumeDim>& element,
    const Direction<VolumeDim>& direction_to_neighbor) noexcept;

}  // namespace Weno_detail
}  // namespace SlopeLimiters
