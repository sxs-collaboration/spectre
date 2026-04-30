// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <type_traits>

#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Tags/LtsStepChoosers.hpp"
#include "Time/Tags/StepperErrorTolerancesCompute.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
struct StepperErrorEstimate;
struct StepperErrorTolerances;
class TimeDelta;
class TimeStepId;
class TimeStepper;
namespace Tags {
template <typename Tag>
struct HistoryEvolvedVariables;
template <typename Tag>
struct Next;
struct StepperErrorEstimatesEnabled;
template <typename Tag>
struct StepperErrorTolerances;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace TimeSteppers {
template <typename Vars>
class History;
}  // namespace TimeSteppers
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// \ingroup TimeGroup
/// \brief Perform variable updates for one substep
/// @{
template <typename System,
          template <typename> typename CacheTagPrefix = std::type_identity_t,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
struct UpdateU;

template <typename System, template <typename> typename CacheTagPrefix,
          typename... VariablesTags>
struct UpdateU<System, CacheTagPrefix, tmpl::list<VariablesTags...>> {
  using const_global_cache_tags =
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                 CacheTagPrefix<::Tags::LtsStepChoosers>,
                 CacheTagPrefix<::Tags::VariableOrderAlgorithm>>;
  using simple_tags = tmpl::list<::Tags::StepperErrors<VariablesTags>...>;
  using compute_tags = tmpl::list<
      Tags::StepperErrorEstimatesEnabledCompute<CacheTagPrefix>,
      Tags::StepperErrorTolerancesCompute<VariablesTags, CacheTagPrefix>...>;

  using return_tags =
      tmpl::list<VariablesTags..., Tags::StepperErrors<VariablesTags>...>;
  using argument_tags =
      tmpl::list<CacheTagPrefix<Tags::TimeStepper<TimeStepper>>,
                 Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 Tags::StepperErrorEstimatesEnabled,
                 Tags::HistoryEvolvedVariables<VariablesTags>...,
                 Tags::StepperErrorTolerances<VariablesTags>...>;

  static void apply(
      const gsl::not_null<typename VariablesTags::type*>... vars,
      const typename tmpl::has_type<
          VariablesTags,
          gsl::not_null<std::array<std::optional<StepperErrorEstimate>, 2>*>>::
          type... errors,
      const TimeStepper& time_stepper, const TimeStepId& time_step_id,
      const TimeStepId& next_time_step_id, const TimeDelta& time_step,
      bool error_estimates_enabled,
      const TimeSteppers::History<typename VariablesTags::type>&... histories,
      const typename tmpl::has_type<
          VariablesTags, StepperErrorTolerances>::type&... tolerances);
};
/// @}
