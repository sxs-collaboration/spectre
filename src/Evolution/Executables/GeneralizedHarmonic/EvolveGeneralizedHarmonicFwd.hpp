// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace GeneralizedHarmonic {
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

template <typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars;
/// \endcond
