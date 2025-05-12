// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

/*!
 * \brief Combine the volume subcell variables and the ghost variables
 * contained as DataVector into a single combined DataVector.
 */

namespace evolution::dg::subcell {
template <size_t Dim>
DataVector combine_volume_ghost_data(const DataVector& volume_data,
                                     const DataVector& ghost_data,
                                     const Index<Dim>& subcell_extents,
                                     size_t ghost_zone_size,
                                     const Direction<Dim>& direction_to_extend);
}  // namespace evolution::dg::subcell
