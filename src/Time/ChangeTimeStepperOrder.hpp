// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/TaggedVariant.hpp"
#include "Time/History.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
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
/// \endcond

namespace ChangeTimeStepperOrder_detail {
// Doxygen is confused by the self-inheritance, even though this is in
// a detail namespace.
/// \cond
template <typename VariablesTag>
struct apply_impl : apply_impl<tmpl::list<VariablesTag>> {};

template <typename... VariablesTags>
struct apply_impl<tmpl::list<VariablesTags...>> {
  static void apply(
      const gsl::not_null<
          TimeSteppers::History<typename VariablesTags::type>*>... histories,
      const TimeStepper& time_stepper,
      const VariableOrderAlgorithm& order_algorithm,
      const TimeStepId& next_time_step_id,
      const typename tmpl::has_type<
          VariablesTags, std::array<std::optional<StepperErrorEstimate>,
                                    2>>::type&... errors) {
    if (next_time_step_id.substep() != 0) {
      return;
    }

    const auto order_variant = time_stepper.order();
    const auto* stepper_order =
        variants::get_if<TimeSteppers::Tags::VariableOrder>(&order_variant);
    if (stepper_order == nullptr) {
      // Running at fixed order.
      return;
    }

    const size_t new_order =
        std::clamp(order_algorithm.template choose_order<VariablesTags...>(
                       *histories..., errors...),
                   stepper_order->minimum, stepper_order->maximum);

    expand_pack((histories->integration_order(new_order), 0)...);
  }
};
/// \endcond
}  // namespace ChangeTimeStepperOrder_detail

/*!
 * \ingroup TimeGroup
 * \brief Adjust the step order for local time-stepping
 *
 * \details See VariableOrderAlgorithm for descriptions of the
 * algorithms.  This mutator wraps that class, checking that the order
 * is not changed out of the valid range for the time stepper.
 */
template <typename System>
struct ChangeTimeStepperOrder : ChangeTimeStepperOrder_detail::apply_impl<
                                    typename System::variables_tag> {
  using variables_tags = tmpl::conditional_t<
      tt::is_a_v<tmpl::list, typename System::variables_tag>,
      typename System::variables_tag,
      tmpl::list<typename System::variables_tag>>;

  using const_global_cache_tags = tmpl::list<Tags::VariableOrderAlgorithm>;
  using return_tags =
      tmpl::transform<variables_tags,
                      tmpl::bind<Tags::HistoryEvolvedVariables, tmpl::_1>>;
  using argument_tags = tmpl::append<
      tmpl::list<Tags::TimeStepper<TimeStepper>, Tags::VariableOrderAlgorithm,
                 Tags::Next<Tags::TimeStepId>>,
      tmpl::transform<variables_tags,
                      tmpl::bind<Tags::StepperErrors, tmpl::_1>>>;

  // apply defined in apply_impl above
};
