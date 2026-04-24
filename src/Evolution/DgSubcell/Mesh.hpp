// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

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
size_t get_computational_dim(const Mesh<Dim>& subcell_mesh) {
  if constexpr (Dim == 3) {
    if (subcell_mesh.quadrature(2) == Spectral::Quadrature::SphericalSymmetry) {
      return 1;
    } else if (subcell_mesh.quadrature(2) ==
               Spectral::Quadrature::AxialSymmetry) {
      return 2;
    } else {
      return 3;
    }
  } else {
    return Dim;
  }
}

/*!
 * \brief Computes the computational dimension from the subcell extents, which
 * can be less than `Dim` when Cartoon bases are used.
 */
template <size_t Dim>
size_t get_computational_dim(const Index<Dim>& subcell_extents) {
  if constexpr (Dim == 3) {
    if (subcell_extents[1] == 1) {
      return 1;
    } else if (subcell_extents[2] == 1) {
      return 2;
    } else {
      return 3;
    }
  } else {
    return Dim;
  }
}

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
                            bool neighbor = false) {
  const std::string neighbor_str = neighbor ? " neighbor" : "";
  if constexpr (Dim == 3) {
    if (subcell_extents[1] == 1) {
      // Checking for spherical symmetry
      ASSERT(
          subcell_extents[0] != 1 and subcell_extents[2] == 1,
          "The" << neighbor_str
                << " subcell extents are neither isotropic nor a valid cartoon "
                   "pattern, got "
                << subcell_extents);
    } else if (subcell_extents[2] == 1) {
      // Checking for axial symmetry
      ASSERT(
          subcell_extents.slice_away(2) == Index<2>(subcell_extents[0]),
          "The" << neighbor_str
                << " subcell extents are neither isotropic nor a valid cartoon "
                   "pattern, got "
                << subcell_extents);
    } else {
      // No cartoon, normal extents
      ASSERT(subcell_extents == Index<Dim>(subcell_extents[0]),
             "The" << neighbor_str << " subcell mesh must be uniform but is "
                   << subcell_extents);
    }
  } else {
    ASSERT(subcell_extents == Index<Dim>(subcell_extents[0]),
           "The" << neighbor_str << " subcell mesh must be uniform but is "
                 << subcell_extents);
  }
}
}  // namespace evolution::dg::subcell::fd
