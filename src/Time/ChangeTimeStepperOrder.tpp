// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/ChangeTimeStepperOrder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/TaggedVariant.hpp"
#include "Time/History.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

template <typename System, typename... VariablesTags>
void ChangeTimeStepperOrder<System, tmpl::list<VariablesTags...>>::apply(
    const gsl::not_null<
        TimeSteppers::History<typename VariablesTags::type>*>... histories,
    const TimeStepper& time_stepper,
    const VariableOrderAlgorithm& order_algorithm,
    const TimeStepId& next_time_step_id,
    const typename tmpl::has_type<
        VariablesTags,
        std::array<std::optional<StepperErrorEstimate>, 2>>::type&... errors) {
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
