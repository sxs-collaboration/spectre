// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace RelativisticEuler {
namespace Solutions {
template <size_t Dim>
class SmoothFlow;
}  // namespace Solutions
}  // namespace RelativisticEuler

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;
