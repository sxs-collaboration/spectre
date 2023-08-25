// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace SpatialDiscretization {
/*!
 * \brief Either the choice of quadrature method to compute integration weights
 * for a spectral or discontinuous Galerkin (DG) method, or the locations of
 * grid points when a finite difference method is used
 *
 * \details The particular choices of Basis and Quadrature determine
 * where the collocation points of a Mesh are located in an Element.  For a
 * spectral or DG method, integrals using \f$N\f$ collocation points with Gauss
 * quadrature are exact to polynomial order \f$p=2N-1\f$. Gauss-Lobatto
 * quadrature is exact only to polynomial order \f$p=2N-3\f$, but includes
 * collocation points at the domain boundary.  For a finite difference
 * method, one needs to choose the order of the scheme (and hence the
 * weights, differentiation matrix, integration weights, and
 * interpolant) locally in space and time to handle discontinuous
 * solutions.
 *
 * \note Choose `Gauss` or `GaussLobatto` when using Basis::Legendre or
 * Basis::Chebyshev.
 *
 * \note Choose `CellCentered` or `FaceCentered` when using
 * Basis::FiniteDifference.
 *
 * \note When using Basis::SphericalHarmonic in consecutive dimensions, choose
 * `Gauss` for the first dimension and `Equiangular` in the second dimension.
 */
enum class Quadrature {
  Gauss,
  GaussLobatto,
  CellCentered,
  FaceCentered,
  Equiangular
};

/// Output operator for a Quadrature.
std::ostream& operator<<(std::ostream& os, const Quadrature& quadrature);
}  // namespace SpatialDiscretization

template <>
struct Options::create_from_yaml<SpatialDiscretization::Quadrature> {
  template <typename Metavariables>
  static SpatialDiscretization::Quadrature create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
SpatialDiscretization::Quadrature
Options::create_from_yaml<SpatialDiscretization::Quadrature>::create<void>(
    const Options::Option& options);
