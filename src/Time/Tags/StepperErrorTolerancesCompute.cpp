// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Tags/StepperErrorTolerancesCompute.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "DataStructures/TaggedVariant.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/FixedLtsRatio.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Tags {
namespace {
template <typename StepChooserUse>
bool requests_any_tolerances(const StepChooser<StepChooserUse>& step_chooser) {
  const auto* const tolerance_request =
      dynamic_cast<const RequestsStepperErrorTolerances*>(&step_chooser);
  return tolerance_request != nullptr and
         not tolerance_request->tolerances().empty();
}
}  // namespace

namespace StepperErrorEstimatesEnabledCompute_detail {
void lts_function(
    const gsl::not_null<bool*> error_estimates_enabled,
    const ::EventsAndTriggers& events_and_triggers,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers) {
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
            if (const auto* const fixed_ratio =
                    dynamic_cast<const ::StepChoosers::FixedLtsRatio*>(
                        &step_chooser);
                fixed_ratio != nullptr) {
              fixed_ratio->for_each_step_chooser(
                  [&](const StepChooser<StepChooserUse::LtsStep>&
                          sub_step_chooser) {
                    if (*error_estimates_enabled) {
                      return;
                    }
                    if (requests_any_tolerances(sub_step_chooser)) {
                      *error_estimates_enabled = true;
                    }
                  });
            }
          });
    }
  });

  for (const auto& step_chooser : step_choosers) {
    if (requests_any_tolerances(*step_chooser)) {
      *error_estimates_enabled = true;
      return;
    }
  }
}

void gts_function(const gsl::not_null<bool*> error_estimates_enabled,
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
            if (requests_any_tolerances(step_chooser)) {
              *error_estimates_enabled = true;
            }
          });
    }
  });
}
}  // namespace StepperErrorEstimatesEnabledCompute_detail

namespace StepperErrorTolerancesCompute_detail {
namespace {
template <typename StepChooserUse>
void set_tolerances_if_requested(
    const gsl::not_null<::StepperErrorTolerances*> tolerances,
    const StepChooser<StepChooserUse>& step_chooser,
    const std::type_index& tag_type, const std::string& tag_name) {
  if (const auto* const tolerance_request =
          dynamic_cast<const RequestsStepperErrorTolerances*>(&step_chooser);
      tolerance_request != nullptr) {
    const auto tolerances_map = tolerance_request->tolerances();
    if (const auto this_tolerances = tolerances_map.find(tag_type);
        this_tolerances != tolerances_map.end()) {
      if (tolerances->estimates != ::StepperErrorTolerances::Estimates::None and
          *tolerances != this_tolerances->second) {
        ERROR_NO_TRACE("All ErrorControl events for "
                       << tag_name << " must use the same tolerances.");
      }
      *tolerances = this_tolerances->second;
    }
  }
}
}  // namespace

void lts_impl(
    const gsl::not_null<::StepperErrorTolerances*> tolerances,
    const ::EventsAndTriggers& events_and_triggers,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers,
    const ::TimeStepper& time_stepper,
    const ::VariableOrderAlgorithm& variable_order_algorithm,
    const std::type_index& tag_type, const std::string& tag_name) {
  *tolerances = ::StepperErrorTolerances{};

  events_and_triggers.for_each_event([&](const auto& event) {
    if (const auto* const change_slab_size =
            dynamic_cast<const ::Events::ChangeSlabSize*>(&event)) {
      change_slab_size->for_each_step_chooser(
          [&](const StepChooser<StepChooserUse::Slab>& step_chooser) {
            if (const auto* const fixed_ratio =
                    dynamic_cast<const ::StepChoosers::FixedLtsRatio*>(
                        &step_chooser);
                fixed_ratio != nullptr) {
              fixed_ratio->for_each_step_chooser(
                  [&](const StepChooser<StepChooserUse::LtsStep>&
                          sub_step_chooser) {
                    set_tolerances_if_requested(tolerances, sub_step_chooser,
                                                tag_type, tag_name);
                  });
            }
          });
    }
  });

  for (const auto& step_chooser : step_choosers) {
    set_tolerances_if_requested(tolerances, *step_chooser, tag_type, tag_name);
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

void gts_impl(const gsl::not_null<::StepperErrorTolerances*> tolerances,
              const ::EventsAndTriggers& events_and_triggers,
              const std::type_index& tag_type, const std::string& tag_name) {
  *tolerances = ::StepperErrorTolerances{};
  // In principle the slab size could be changed based on a dense
  // trigger, but it's not clear that there is ever a good reason to
  // do so, and it wouldn't make sense to use error control in that
  // context in any case.
  events_and_triggers.for_each_event([&](const auto& event) {
    if (const auto* const change_slab_size =
            dynamic_cast<const ::Events::ChangeSlabSize*>(&event)) {
      change_slab_size->for_each_step_chooser(
          [&](const StepChooser<StepChooserUse::Slab>& step_chooser) {
            set_tolerances_if_requested(tolerances, step_chooser, tag_type,
                                        tag_name);
          });
    }
  });
}
}  // namespace StepperErrorTolerancesCompute_detail
}  // namespace Tags
