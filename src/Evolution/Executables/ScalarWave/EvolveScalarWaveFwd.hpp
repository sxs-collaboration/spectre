// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace ScalarWave {
namespace Solutions {
class RegularSphericalWave;
template <size_t Dim>
class PlaneWave;
}  // namespace Solutions
}  // namespace ScalarWave

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;
/// \endcond
