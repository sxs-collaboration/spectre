// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

namespace Limiters {
namespace Weno_detail {

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
double oscillation_indicator(const DataVector& data,
                             const Mesh<VolumeDim>& mesh) noexcept;

}  // namespace Weno_detail
}  // namespace Limiters
