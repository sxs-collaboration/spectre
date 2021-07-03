// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
class TovSolution;
}  // namespace Solutions
}  // namespace gr

namespace RelativisticEuler {
namespace Solutions {
class FishboneMoncriefDisk;
}  // namespace Solutions
}  // namespace RelativisticEuler

namespace grmhd {
namespace Solutions {
class BondiMichel;
}  // namespace Solutions
namespace AnalyticData {
class BondiHoyleAccretion;
class MagnetizedFmDisk;
}  // namespace AnalyticData
}  // namespace grmhd

struct KerrHorizon;
template <typename InitialData, typename...InterpolationTargetTags>
struct EvolutionMetavars;
/// \endcond
