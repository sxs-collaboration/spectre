// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/PhaseControl/PhaseControlTags.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"

namespace PhaseControl {
TriggerAndPhaseChanges::TriggerAndPhaseChanges() = default;
TriggerAndPhaseChanges::TriggerAndPhaseChanges(
    std::unique_ptr<::Trigger> trigger_in,
    std::vector<std::unique_ptr<::PhaseChange>> phase_changes_in)
    : trigger(std::move(trigger_in)),
      phase_changes(std::move(phase_changes_in)) {}

void TriggerAndPhaseChanges::pup(PUP::er& p) {
  p | trigger;
  p | phase_changes;
}
}  // namespace PhaseControl
