// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell {
// @{
/// Set `status` to `1` if the cell is marked as needing subcell, otherwise set
/// `status` to `0`. `status` has grid points on the currently active mesh.
///
/// \note It is possible to encounter a `status` of `0`, indicating a smooth
/// solution, even though the active mesh is the subcell mesh. This can occur
/// when using a multistep time integration method like Adams Bashforth, where
/// the FD/FV scheme is used as long as _any_ of the timestepper history is
/// flagged for FD/FV subcell evolution. An example of this would be just after
/// a shock leaves the cell: the solution is now smooth (`status` takes value
/// `0`), but the multistep method continues to take FD timesteps (`status` is
/// defined on the subcell mesh) until the entire history is flagged as smooth.
/// Thus, `tci_history.front()` (which corresponds to the most recent TCI
/// status) is used to set the status when using multistep time integrators,
/// while the `active_grid` is used when using other time steppers.
template <size_t Dim>
void tci_status(gsl::not_null<Scalar<DataVector>*> status,
                const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
                subcell::ActiveGrid active_grid,
                const std::deque<subcell::ActiveGrid>& tci_history) noexcept;

template <size_t Dim>
Scalar<DataVector> tci_status(
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    subcell::ActiveGrid active_grid,
    const std::deque<subcell::ActiveGrid>& tci_history) noexcept;
// @}
}  // namespace evolution::dg::subcell
