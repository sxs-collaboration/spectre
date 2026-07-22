// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

/// Limits on the extents of a Mesh based on Basis and Quadrature.  The minima
/// are strict requirements except for spherical harmonics which are set by the
/// package (SPHEREPACK) used to implement them.  The maxima are set to limit
/// the sizes of static caches, so can be changed if desired.
namespace Spectral::limits {
/// The minimum allowed mode \f$L\f$ for SPHEREPACK, the package used for a
/// spherical harmonic basis
static constexpr size_t min_spherical_harmonic_mode = 2;

/// The maximum allowed mode \f$L\f$ for spherical harmonics in S2 or B3
/// topologies
static constexpr size_t max_spherical_harmonic_mode = 40;

/// The maximum allowed mode \f$M\f$ for a Fourier series in S1 or B2 topologies
static constexpr size_t max_fourier_mode = 40;

/// The maximum allowed mode \f$N\f$ for polynomial series in I1 or B1
/// topologies
static constexpr size_t max_i1_polynomial_mode = 20;

/// \brief The maximum allowed extent for a given Basis and Quadrature
///
/// \details The maxima are chosen to limit the size of static caches.
/// Let \f$N\f$ be the highest represented polynomial mode, \f$L\f$ the highest
/// represented spherical harmonic mode, and \f$M\f$ the highest represented
/// Fourier mode.  Then the maxima are set as follows:
/// - \f$N+1\f$ for Basis::Legendre, Basis::Chebyshev and Basis::ZernikeB1
/// - \f$2N+2\f$ for Basis::FiniteDifference as for DG-subcell
///   \f$n_{FD} = 2 n_{DG}\f$ for Quadrature::FaceCentered
/// - \f$(L+1, 2L+1)\f$ for Basis::SphericalHarmonic
/// - \f$2M+1\f$ for Basis::Fourier
/// - \f$(M/2 + 1, 2M+1)\f$ for Basis::ZernikeB2
/// - \f$(L/2 + 1, L+1, 2L+1)\f$ for Basis::ZernikeB3
/// - 1 for Basis::Cartoon
constexpr size_t max(const Basis basis, const Quadrature quadrature) {
  if (basis == Basis::Legendre or basis == Basis::Chebyshev or
      basis == Basis::ZernikeB1) {
    return max_i1_polynomial_mode + 1;
  } else if (basis == Basis::FiniteDifference) {
    return 2 * max_i1_polynomial_mode + 2;
  } else if (basis == Basis::Cartoon) {
    return 1;
  } else if (basis == Basis::SphericalHarmonic or basis == Basis::ZernikeB3) {
    if (quadrature == Quadrature::Gauss) {
      return max_spherical_harmonic_mode + 1;
    } else if (quadrature == Quadrature::Equiangular) {
      return 2 * max_spherical_harmonic_mode + 1;
    } else if (quadrature == Quadrature::GaussRadauUpper) {
      return max_spherical_harmonic_mode / 2 + 1;
    }
  } else if (basis == Basis::ZernikeB2 and
             quadrature == Quadrature::GaussRadauUpper) {
    return max_fourier_mode / 2 + 1;
  } else if (quadrature == Quadrature::Equiangular) {
    return 2 * max_fourier_mode + 1;
  }
  return 0;
}

/// \brief The minimum extent for a given Basis and Quadrature
///
/// \details The minimum is 1 except for the following:
/// - 2 for Quadrature::GaussLobatto and Quadrature::FaceCentered as they have
///   collocation points on the boundaries
/// - \f$(3, 5)\f$ for the angular directions of Basis::SphericalHarmonic or
///   \f$(2, 3, 5)\f$ for the directions of Basis::ZernikeB3 as this is a
///   requirement of the package SPHEREPACK that implements spherical harmonics
constexpr size_t min(const Basis basis, const Quadrature quadrature) {
  if (quadrature == Quadrature::GaussLobatto or
      quadrature == Quadrature::FaceCentered or
      (basis == Basis::ZernikeB3 and
       quadrature == Quadrature::GaussRadauUpper)) {
    return 2;
  } else if (basis == Basis::SphericalHarmonic or basis == Basis::ZernikeB3) {
    if (quadrature == Quadrature::Gauss) {
      return min_spherical_harmonic_mode + 1;
    } else if (quadrature == Quadrature::Equiangular) {
      return 2 * min_spherical_harmonic_mode + 1;
    }
  }
  return 1;
}
}  // namespace Spectral::limits
