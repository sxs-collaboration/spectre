// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/ChangeSlabSize/Event.hpp"

#include <typeindex>
#include <unordered_map>

#include "Time/CollectStepperErrorTolerances.hpp"
#include "Time/StepperErrorTolerances.hpp"

namespace Events {
std::unordered_map<std::type_index, StepperErrorTolerances>
ChangeSlabSize::tolerances() const {
  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
  for (const auto& step_chooser : step_choosers_) {
    collect_stepper_error_tolerances(&tolerances, *step_chooser);
  }
  return tolerances;
}

PUP::able::PUP_ID ChangeSlabSize::my_PUP_ID = 0;  // NOLINT
}  // namespace Events
