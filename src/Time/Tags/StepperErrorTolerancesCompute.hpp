// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
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
struct LtsStepChoosers;
template <typename StepperInterface>
struct TimeStepper;
struct VariableOrderAlgorithm;
}  // namespace Tags
/// \endcond

namespace Tags {
namespace StepperErrorEstimatesEnabledCompute_detail {
void function(
    gsl::not_null<bool*> error_estimates_enabled,
    const ::EventsAndTriggers& events_and_triggers,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers);
}  // namespace StepperErrorEstimatesEnabledCompute_detail

/// \ingroup TimeGroup
/// \brief Searches the StepChoosers for any requesting error estimates.
template <template <typename> typename CacheTagPrefix = std::type_identity_t>
struct StepperErrorEstimatesEnabledCompute : db::ComputeTag,
                                             StepperErrorEstimatesEnabled {
  using base = StepperErrorEstimatesEnabled;
  using return_type = type;
  using argument_tags =
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                 CacheTagPrefix<::Tags::LtsStepChoosers>>;

  static constexpr auto function =
      &StepperErrorEstimatesEnabledCompute_detail::function;
};

namespace StepperErrorTolerancesCompute_detail {
void function(
    gsl::not_null<::StepperErrorTolerances*> tolerances,
    const ::EventsAndTriggers& events_and_triggers,
    const std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
        step_choosers,
    const ::TimeStepper& time_stepper,
    const ::VariableOrderAlgorithm& variable_order_algorithm,
    const std::type_index& tag_type);
}  // namespace StepperErrorTolerancesCompute_detail

/// \ingroup TimeGroup
/// \brief A tag that contains the error tolerances if any StepChooser
/// requests an error estimate for the variable.
template <typename EvolvedVariableTag,
          template <typename> typename CacheTagPrefix = std::type_identity_t>
struct StepperErrorTolerancesCompute
    : db::ComputeTag,
      StepperErrorTolerances<EvolvedVariableTag> {
  using base = StepperErrorTolerances<EvolvedVariableTag>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                 CacheTagPrefix<::Tags::LtsStepChoosers>,
                 CacheTagPrefix<::Tags::TimeStepper<::TimeStepper>>,
                 CacheTagPrefix<::Tags::VariableOrderAlgorithm>>;

  static void function(
      const gsl::not_null<::StepperErrorTolerances*> tolerances,
      const ::EventsAndTriggers& events_and_triggers,
      const std::vector<
          std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
          step_choosers,
      const ::TimeStepper& time_stepper,
      const ::VariableOrderAlgorithm& variable_order_algorithm) {
    StepperErrorTolerancesCompute_detail::function(
        tolerances, events_and_triggers, step_choosers, time_stepper,
        variable_order_algorithm, typeid(EvolvedVariableTag));
  }
};
}  // namespace Tags
