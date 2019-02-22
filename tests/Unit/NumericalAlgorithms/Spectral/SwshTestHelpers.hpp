// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "NumericalAlgorithms/Spectral/SwshTags.hpp"  // IWYU pragma: keep

namespace Spectral {
namespace Swsh {
namespace TestHelpers {

// returns the factorial of the argument as a double so that an approximate
// value can be given for larger input quantities. Note that the spin-weighted
// harmonic function requires a factorial of l + m, so harmonics above l~12
// would be untestable if the factorial only returned size_t's.
double factorial(size_t arg) noexcept;

// Note that the methods for computing the spin-weighted spherical harmonics and
// their derivatives below are both 1) poorly optimized (they use many
// computations per grid point evaluated) and 2) not terribly accurate (the
// analytic expressions require evaluation of ratios of factorials, losing
// numerical precision rapidly). However, they are comparatively easy to
// manually check for correctness, which is critical to offer a reliable measure
// of the spin-weighted transforms.

// Analytic form for the spin-weighted spherical harmonic function, for testing
// purposes. The formula is from [Goldberg
// et. al.](https://aip.scitation.org/doi/10.1063/1.1705135)
std::complex<double> spin_weighted_spherical_harmonic(int s, int l, int m,
                                                      double theta,
                                                      double phi) noexcept;

// Returns the value of the spin-weighted derivative `DerivativeKind` of the
// spherical harmonic basis function \f${}_s Y_{l m}\f$ at angular location
// (`theta`, `phi`) using the recurrence identities for spin-weighted
// derivatives.
template <typename DerivativeKind>
std::complex<double> derivative_of_spin_weighted_spherical_harmonic(
    int s, int l, int m, double theta, double phi) noexcept;
}  // namespace TestHelpers
}  // namespace Swsh
}  // namespace Spectral
