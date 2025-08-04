// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Tags/StepperErrorTolerancesCompute.hpp"

#include <memory>
#include <vector>

#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Utilities/Gsl.hpp"

namespace Tags {
template <>
void StepperErrorEstimatesEnabledCompute<true>::function(
    const gsl::not_null<bool*> error_estimates_enabled,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers) {
  *error_estimates_enabled = false;
  for (const auto& step_chooser : step_choosers) {
    if (dynamic_cast<const RequestsAnyStepperErrorTolerances*>(
            &*step_chooser) != nullptr) {
      *error_estimates_enabled = true;
      return;
    }
  }
}

template <>
void StepperErrorEstimatesEnabledCompute<false>::function(
    const gsl::not_null<bool*> error_estimates_enabled,
    const ::EventsAndTriggers& events_and_triggers) {
  // In principle the slab size could be changed based on a dense
  // trigger, but it's not clear that there is ever a good reason to
  // do so, and it wouldn't make sense to use error control in that
  // context in any case.
  *error_estimates_enabled = false;
  events_and_triggers.for_each_event([&](const auto& event) {
    if (*error_estimates_enabled) {
      return;
    }
    if (const auto* const change_slab_size =
            dynamic_cast<const ::Events::ChangeSlabSize*>(&event)) {
      change_slab_size->for_each_step_chooser(
          [&](const StepChooser<StepChooserUse::Slab>& step_chooser) {
            if (*error_estimates_enabled) {
              return;
            }
            if (dynamic_cast<const RequestsAnyStepperErrorTolerances*>(
                    &step_chooser) != nullptr) {
              *error_estimates_enabled = true;
            }
          });
    }
  });
}
}  // namespace Tags
