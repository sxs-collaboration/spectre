// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace gh {
template <size_t Dim>
struct System;
namespace Solutions {
template <typename GrSolution>
struct WrappedGr;
}  // namespace Solutions
}  // namespace gh
namespace gr {
namespace Solutions {
template <size_t Dim>
struct GaugeWave;
}  // namespace Solutions
}  // namespace gr

template <size_t VolumeDim, bool EvolveCcm>
struct EvolutionMetavars;
/// \endcond
