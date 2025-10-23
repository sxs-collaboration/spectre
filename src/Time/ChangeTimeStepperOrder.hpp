// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <type_traits>

#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
struct StepperErrorEstimate;
class TimeStepId;
class TimeStepper;
class VariableOrderAlgorithm;
namespace Tags {
template <typename Tag>
struct HistoryEvolvedVariables;
template <typename Tag>
struct Next;
template <typename Tag>
struct StepperErrors;
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

/*!
 * \ingroup TimeGroup
 * \brief Adjust the step order for local time-stepping
 *
 * \details See VariableOrderAlgorithm for descriptions of the
 * algorithms.  This mutator wraps that class, checking that the order
 * is not changed out of the valid range for the time stepper.
 */
/// @{
template <typename System,
          template <typename> typename CacheTagPrefix = std::type_identity_t,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
struct ChangeTimeStepperOrder;

template <typename System, template <typename> typename CacheTagPrefix,
          typename... VariablesTags>
struct ChangeTimeStepperOrder<System, CacheTagPrefix,
                              tmpl::list<VariablesTags...>> {
  using const_global_cache_tags = tmpl::list<Tags::VariableOrderAlgorithm>;
  using return_tags =
      tmpl::list<Tags::HistoryEvolvedVariables<VariablesTags>...>;
  using argument_tags =
      tmpl::list<CacheTagPrefix<Tags::TimeStepper<TimeStepper>>,
                 Tags::VariableOrderAlgorithm, Tags::Next<Tags::TimeStepId>,
                 Tags::StepperErrors<VariablesTags>...>;

  static void apply(
      const gsl::not_null<
          TimeSteppers::History<typename VariablesTags::type>*>... histories,
      const TimeStepper& time_stepper,
      const VariableOrderAlgorithm& order_algorithm,
      const TimeStepId& next_time_step_id,
      const typename tmpl::has_type<
          VariablesTags,
          std::array<std::optional<StepperErrorEstimate>, 2>>::type&... errors);
};
/// @}
