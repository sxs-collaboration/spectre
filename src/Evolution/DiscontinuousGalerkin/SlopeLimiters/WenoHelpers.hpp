// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

// IWYU pragma: no_forward_declare Variables

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

namespace SlopeLimiters {
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
//
// Where (this reference to be added to References.bib later, when it is cited
// from _rendered_ documentation):
// Dumbser2007:
//   Dumbser, M and Kaeser, M
//   Arbitrary high order non-oscillatory finite volume schemes on unstructured
//   meshes for linear hyperbolic systems
//   https://doi.org/10.1016/j.jcp.2006.06.043
template <size_t VolumeDim>
double oscillation_indicator(const DataVector& data,
                             const Mesh<VolumeDim>& mesh) noexcept;

}  // namespace Weno_detail
}  // namespace SlopeLimiters
