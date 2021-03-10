// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::fd {
/*!
 * \brief Computes the cell-centered finite-difference mesh from the DG mesh,
 * using \f$2N-1\f$ grid points per dimension, where \f$N\f$ is the degree of
 * the DG basis.
 */
template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& dg_mesh) noexcept;
}  // namespace evolution::dg::subcell::fd
