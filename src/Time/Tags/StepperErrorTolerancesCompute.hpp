// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/TaggedVariant.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <Triggers::WhenToCheck WhenToCheck>
struct EventsAndTriggers;
struct StepChoosers;
template <typename StepperInterface>
struct TimeStepper;
struct VariableOrderAlgorithm;
}  // namespace Tags
/// \endcond

namespace Tags {
/// \ingroup TimeGroup
/// \brief Searches the StepChoosers for any requesting error estimates.
template <bool LocalTimeStepping>
struct StepperErrorEstimatesEnabledCompute : db::ComputeTag,
                                             StepperErrorEstimatesEnabled {
  using base = StepperErrorEstimatesEnabled;
  using return_type = type;
  using argument_tags = tmpl::conditional_t<
      LocalTimeStepping, tmpl::list<::Tags::StepChoosers>,
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>>;

  // local time stepping
  static void function(
      gsl::not_null<bool*> error_estimates_enabled,
      const std::vector<
          std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
          step_choosers);

  // global time stepping
  static void function(gsl::not_null<bool*> error_estimates_enabled,
                       const ::EventsAndTriggers& events_and_triggers);
};

/// \ingroup TimeGroup
/// \brief A tag that contains the error tolerances if any StepChooser
/// requests an error estimate for the variable.
template <typename EvolvedVariableTag, bool LocalTimeStepping>
struct StepperErrorTolerancesCompute
    : db::ComputeTag,
      StepperErrorTolerances<EvolvedVariableTag> {
  using base = StepperErrorTolerances<EvolvedVariableTag>;
  using return_type = typename base::type;
  using argument_tags = tmpl::conditional_t<
      LocalTimeStepping,
      tmpl::list<::Tags::StepChoosers, ::Tags::TimeStepper<::TimeStepper>,
                 ::Tags::VariableOrderAlgorithm>,
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>>;

  // local time stepping
  static void function(
      const gsl::not_null<::StepperErrorTolerances*> tolerances,
      const std::vector<
          std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
          step_choosers,
      const ::TimeStepper& time_stepper,
      const ::VariableOrderAlgorithm& variable_order_algorithm) {
    *tolerances = ::StepperErrorTolerances{};
    for (const auto& step_chooser : step_choosers) {
      set_tolerances_if_requested(tolerances, *step_chooser);
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

  // global time stepping
  static void function(
      const gsl::not_null<::StepperErrorTolerances*> tolerances,
      const ::EventsAndTriggers& events_and_triggers) {
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
              set_tolerances_if_requested(tolerances, step_chooser);
            });
      }
    });
  }

 private:
  template <typename StepChooserUse>
  static void set_tolerances_if_requested(
      const gsl::not_null<::StepperErrorTolerances*> tolerances,
      const StepChooser<StepChooserUse>& step_chooser) {
    if (const auto* const tolerance_request = dynamic_cast<
            const RequestsStepperErrorTolerances<EvolvedVariableTag>*>(
            &step_chooser);
        tolerance_request != nullptr) {
      const auto this_tolerances = tolerance_request->tolerances();
      if (tolerances->estimates != ::StepperErrorTolerances::Estimates::None and
          *tolerances != this_tolerances) {
        ERROR_NO_TRACE("All ErrorControl events for "
                       << db::tag_name<EvolvedVariableTag>()
                       << " must use the same tolerances.");
      }
      *tolerances = this_tolerances;
    }
  }
};
}  // namespace Tags
