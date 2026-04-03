// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/PhaseControl/CheckpointAndExitIfComplete.hpp"

namespace gh::bbh::phase_control {
void CheckpointAndExitIfComplete::pup(PUP::er& p) { PhaseChange::pup(p); }

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID CheckpointAndExitIfComplete::my_PUP_ID = 0;  // NOLINT
#endif  // SPECTRE_USE_CHARM
}  // namespace gh::bbh::phase_control
