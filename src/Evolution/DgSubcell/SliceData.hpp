// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace evolution::dg::subcell {
namespace detail {
template <size_t Dim>
DirectionMap<Dim, std::vector<double>> slice_data_impl(
    const gsl::span<const double>& volume_subcell_vars,
    const Index<Dim>& subcell_extents, size_t number_of_ghost_points,
    const DirectionMap<Dim, bool>& directions_to_slice);
}  // namespace detail

/*!
 * \brief Slice the subcell variables needed for neighbors to perform
 * reconstruction.
 *
 * Note that we slice to a grid that is against the boundary of the element but
 * is several ghost points deep. This is in contrast to the slicing used in the
 * DG method which is to the boundary of the element only.
 *
 * The `number_of_ghost_points` will depend on the number of neighboring points
 * the reconstruction method needs that is used on the subcell. The
 * `directions_to_slice` determines in which directions data is sliced.
 * Generally this will be the directions in which the element has neighbors.
 *
 * The data always has the same ordering as the volume data (tags have the same
 * ordering, grid points are x-varies-fastest).
 */
template <size_t Dim, typename TagList>
DirectionMap<Dim, std::vector<double>> slice_data(
    const Variables<TagList>& volume_subcell_vars,
    const Index<Dim>& subcell_extents, const size_t number_of_ghost_points,
    const DirectionMap<Dim, bool>& directions_to_slice) {
  return detail::slice_data_impl(
      gsl::make_span(volume_subcell_vars.data(), volume_subcell_vars.size()),
      subcell_extents, number_of_ghost_points, directions_to_slice);
}
}  // namespace evolution::dg::subcell
