// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class EventsAndTriggers;
template <typename StepChooserUse>
class StepChooser;
struct StepperErrorTolerances;
class TimeStepper;
class VariableOrderAlgorithm;
namespace StepChooserUse {
struct LtsStep;
}  // namespace StepChooserUse
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

namespace StepperErrorTolerancesCompute_detail {
void lts_impl(
    gsl::not_null<::StepperErrorTolerances*> tolerances,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers,
    const ::TimeStepper& time_stepper,
    const ::VariableOrderAlgorithm& variable_order_algorithm,
    const std::type_index& tag_type, const std::string& tag_name);

void gts_impl(gsl::not_null<::StepperErrorTolerances*> tolerances,
              const ::EventsAndTriggers& events_and_triggers,
              const std::type_index& tag_type, const std::string& tag_name);
}  // namespace StepperErrorTolerancesCompute_detail

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
    StepperErrorTolerancesCompute_detail::lts_impl(
        tolerances, step_choosers, time_stepper, variable_order_algorithm,
        typeid(EvolvedVariableTag), db::tag_name<EvolvedVariableTag>());
  }

  // global time stepping
  static void function(
      const gsl::not_null<::StepperErrorTolerances*> tolerances,
      const ::EventsAndTriggers& events_and_triggers) {
    StepperErrorTolerancesCompute_detail::gts_impl(
        tolerances, events_and_triggers, typeid(EvolvedVariableTag),
        db::tag_name<EvolvedVariableTag>());
  }
};
}  // namespace Tags
