// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
namespace RelativisticEuler {
namespace Solutions {
class FishboneMoncriefDisk;
class RotatingStar;
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
class CcsnCollapse;
class KhInstability;
class MagneticFieldLoop;
class MagneticRotor;
class MagnetizedFmDisk;
class MagnetizedTovStar;
class OrszagTangVortex;
class RiemannProblem;
class SlabJet;
}  // namespace AnalyticData
}  // namespace grmhd

struct CenterOfStar;
struct KerrHorizon;
template <typename InitialData, typename...InterpolationTargetTags>
struct EvolutionMetavars;
/// \endcond
