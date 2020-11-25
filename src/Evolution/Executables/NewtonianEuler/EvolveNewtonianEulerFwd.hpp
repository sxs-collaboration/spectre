// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace NewtonianEuler {
namespace AnalyticData {
template <size_t Dim>
class KhInstability;
}  // namespace AnalyticData

namespace Solutions {
template <size_t Dim>
class IsentropicVortex;
class LaneEmdenStar;
template <size_t Dim>
class RiemannProblem;
template <size_t Dim>
class SmoothFlow;
}  // namespace Solutions
}  // namespace NewtonianEuler

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;
