// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

namespace Limiters {
namespace Weno_detail {

// Denote different schemes for computing related oscillation indicators by
// changing the relative weight given to each derivative of the input data.
// - Unity: l'th derivative has weight 1
// - PowTwoEll: l'th derivative has weight 2^(2 l - 1)
//   This penalizes higher derivatives more strongly: w(l=4) = 128
// - PowTwoEllOverEllFactorial: l'th derivative has weight 2^(2 l - 1) / (l!)^2
//   This penalizes the 1st and 2nd derivatives most strongly, but then higher
//   derivatives have decreasing weights so are weakly penalized: w(l=4) = 0.222
enum class DerivativeWeight { Unity, PowTwoEll, PowTwoEllOverEllFactorial };

std::ostream& operator<<(std::ostream& os,
                         DerivativeWeight derivative_weight) noexcept;

// Compute the WENO oscillation indicator (also called the smoothness indicator)
//
// The oscillation indicator measures the amount of variation in the input data,
// with larger indicator values corresponding to a larger amount of variation
// (either from large monotonic slopes or from oscillations).
//
// Implements an indicator similar to that of Eq. 23 of Dumbser2007, but with
// the necessary adaptations for use on square/cube grids. We favor this
// indicator because it is formulated in the reference coordinates, which we
// use for the WENO reconstruction, and because it lends itself to an efficient
// implementation.
template <size_t VolumeDim>
double oscillation_indicator(DerivativeWeight derivative_weight,
                             const DataVector& data,
                             const Mesh<VolumeDim>& mesh) noexcept;

}  // namespace Weno_detail
}  // namespace Limiters
