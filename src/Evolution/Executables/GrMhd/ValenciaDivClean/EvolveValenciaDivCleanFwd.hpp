// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
namespace gr {
namespace Solutions {
class TovSolution;
}  // namespace Solutions
}  // namespace gr

namespace RelativisticEuler {
namespace Solutions {
class FishboneMoncriefDisk;
template <typename RadialSolution>
class TovStar;
}  // namespace Solutions
}  // namespace RelativisticEuler

namespace grmhd {
namespace Solutions {
class AlfvenWave;
class BondiMichel;
class KomissarovShock;
class SmoothFlow;
}  // namespace Solutions
namespace AnalyticData {
class BlastWave;
class BondiHoyleAccretion;
class MagneticFieldLoop;
class MagneticRotor;
class MagnetizedFmDisk;
class OrszagTangVortex;
class RiemannProblem;
}  // namespace AnalyticData
}  // namespace grmhd

struct KerrHorizon;
template <typename InitialData, typename...InterpolationTargetTags>
struct EvolutionMetavars;
/// \endcond
