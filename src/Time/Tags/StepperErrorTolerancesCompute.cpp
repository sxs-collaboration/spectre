// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Tags/StepperErrorTolerancesCompute.hpp"

#include <algorithm>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "DataStructures/TaggedVariant.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Time/CollectStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Tags {
namespace StepperErrorEstimatesEnabledCompute_detail {
void function(
    const gsl::not_null<bool*> error_estimates_enabled,
    const ::EventsAndTriggers& events_and_triggers,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers) {
  *error_estimates_enabled = false;

  events_and_triggers.for_each_event([&](const auto& event) {
    if (*error_estimates_enabled) {
      return;
    }

    std::unordered_map<std::type_index, ::StepperErrorTolerances> tolerances{};
    collect_stepper_error_tolerances(&tolerances, event);
    if (not tolerances.empty()) {
      *error_estimates_enabled = true;
    }
  });

  if (*error_estimates_enabled) {
    return;
  }

  for (const auto& step_chooser : step_choosers) {
    std::unordered_map<std::type_index, ::StepperErrorTolerances> tolerances{};
    collect_stepper_error_tolerances(&tolerances, *step_chooser);
    if (not tolerances.empty()) {
      *error_estimates_enabled = true;
      return;
    }
  }
}
}  // namespace StepperErrorEstimatesEnabledCompute_detail

namespace StepperErrorTolerancesCompute_detail {
void function(
    const gsl::not_null<::StepperErrorTolerances*> tolerances,
    const ::EventsAndTriggers& events_and_triggers,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers,
    const ::TimeStepper& time_stepper,
    const ::VariableOrderAlgorithm& variable_order_algorithm,
    const std::type_index& tag_type) {
  std::unordered_map<std::type_index, ::StepperErrorTolerances>
      all_tolerances{};

  // In principle the slab size could be changed based on a dense
  // trigger, but it's not clear that there is ever a good reason to
  // do so, and it wouldn't make sense to use error control in that
  // context in any case.
  events_and_triggers.for_each_event([&](const auto& event) {
    collect_stepper_error_tolerances(&all_tolerances, event);
  });

  for (const auto& step_chooser : step_choosers) {
    collect_stepper_error_tolerances(&all_tolerances, *step_chooser);
  }

  if (const auto this_tolerance = all_tolerances.find(tag_type);
      this_tolerance != all_tolerances.end()) {
    *tolerances = this_tolerance->second;
  } else {
    *tolerances = ::StepperErrorTolerances{};
  }

  // Error-based variable-order requires some variable to be
  // controlled using error control, but in a split-variable system
  // a different variable might be controlled, so no control on this
  // variable is not an error.
  if (tolerances->estimates != ::StepperErrorTolerances::Estimates::None and
      variants::holds_alternative<TimeSteppers::Tags::VariableOrder>(
          time_stepper.order())) {
    tolerances->estimates = std::max(
        tolerances->estimates, variable_order_algorithm.required_estimates());
  }
}
}  // namespace StepperErrorTolerancesCompute_detail
}  // namespace Tags
