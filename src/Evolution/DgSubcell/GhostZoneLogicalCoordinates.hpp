// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <size_t Dim>
class Direction;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell::fd {
/// @{
/*!
 * \brief Computes the logical coordinates of ghost grid points for a given
 * direction and ghost zone size.
 *
 * Let `d` be the axis dimension of the `direction`. The returned coordinate has
 * extents that is same as the volume mesh extents but `[d]`-th value
 * replaced by the ghost zone size.
 *
 * For instance if the (volume) subcell mesh has extents \f$(6,6,6)\f$, ghost
 * zone size is 2, and the `direction` is along Xi axis, the resulting
 * coordinates computed by this function has extents \f$(2,6,6)\f$.
 *
 */
template <size_t Dim>
void ghost_zone_logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
        ghost_logical_coords,
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size,
    const Direction<Dim>& direction);

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> ghost_zone_logical_coordinates(
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size,
    const Direction<Dim>& direction);
/// @}
}  // namespace evolution::dg::subcell::fd
