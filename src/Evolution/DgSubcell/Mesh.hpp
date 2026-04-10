// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <size_t Dim>
class Index;
/// \endcond

namespace evolution::dg::subcell::fd {
/*!
 * \brief Computes the cell-centered finite-difference mesh from the DG mesh,
 * using \f$2N-1\f$ grid points per dimension, where \f$N\f$ is the degree of
 * the DG basis.
 */
template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& dg_mesh);

/*!
 * \brief Computes the DG mesh from the cell-centered finite-difference mesh.
 */
template <size_t Dim>
Mesh<Dim> dg_mesh(const Mesh<Dim>& subcell_mesh, Spectral::Basis basis,
                  Spectral::Quadrature quadrature);

/*!
 * \brief Computes the computational dimension from the subcell mesh, which
 * can be less than `Dim` when Cartoon bases are used.
 */
template <size_t Dim>
size_t get_computational_dim(const Mesh<Dim>& subcell_mesh);

/*!
 * \brief Computes the computational dimension from the subcell extents, which
 * can be less than `Dim` when Cartoon bases are used.
 */
template <size_t Dim>
size_t get_computational_dim(const Index<Dim>& subcell_extents);

/*!
 * \brief Verifies the passed subcell mesh is valid, i.e. properly using
 * Cartoon bases and isotropic in non-Cartoon extents.
 *
 * The \p neighbor argument should be set to `true` when checking a neighbor's
 * mesh (only the output of the assert is modified).
 */
template <size_t Dim>
void verify_subcell_mesh(const Mesh<Dim>& subcell_mesh, bool neighbor = false);

/*!
 * \brief Verifies the passed subcell extents are valid, i.e. fully isotropic
 * or deviating in a Cartoon-specific manner.
 *
 * The \p neighbor argument should be set to `true` when checking a neighbor's
 * mesh (only the output of the assert is modified).
 */
template <size_t Dim>
void verify_subcell_extents(const Index<Dim>& subcell_extents,
                            bool neighbor = false);
}  // namespace evolution::dg::subcell::fd
