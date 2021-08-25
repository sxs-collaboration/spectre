// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace GeneralizedHarmonic {
template <size_t Dim>
struct System;
namespace Solutions {
template <typename GrSolution>
struct WrappedGr;
}  // namespace Solutions
}  // namespace GeneralizedHarmonic
namespace gr {
namespace Solutions {
struct KerrSchild;
}  // namespace Solutions
}  // namespace gr
namespace evolution {
struct NumericInitialData;
}  // namespace evolution

template <size_t VolumeDim, typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars;
/// \endcond
