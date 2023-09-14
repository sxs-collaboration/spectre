// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
namespace gh {
template <size_t Dim>
struct System;
namespace Solutions {
template <typename GrSolution>
struct WrappedGr;
}  // namespace Solutions
}  // namespace gh

namespace RelativisticEuler::Solutions {
class FishboneMoncriefDisk;
}  // namespace RelativisticEuler::Solutions

namespace grmhd {
namespace Solutions {
class BondiMichel;
}  // namespace Solutions
namespace AnalyticData {
class BondiHoyleAccretion;
class CcsnCollapse;
class MagnetizedFmDisk;
}  // namespace AnalyticData
}  // namespace grmhd

struct KerrHorizon;
struct BondiSachs;
template <bool UseControlSystems, typename... InterpolationTargetTags>
struct EvolutionMetavars;
/// \endcond
