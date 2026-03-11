// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Bbh/PhaseControl/CheckpointAndExitIfComplete.hpp"

namespace gh::bbh::phase_control {
void CheckpointAndExitIfComplete::pup(PUP::er& p) { PhaseChange::pup(p); }

PUP::able::PUP_ID CheckpointAndExitIfComplete::my_PUP_ID = 0;  // NOLINT
}  // namespace gh::bbh::phase_control
