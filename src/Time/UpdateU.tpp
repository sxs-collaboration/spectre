// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/UpdateU.hpp"

#include <array>
#include <optional>

#include "DataStructures/TaggedVariant.hpp"
#include "Time/History.hpp"
#include "Time/SelfStart.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

template <typename System, bool LocalTimeStepping, typename... VariablesTags>
void UpdateU<System, LocalTimeStepping, tmpl::list<VariablesTags...>>::apply(
    const gsl::not_null<typename VariablesTags::type*>... vars,
    const typename tmpl::has_type<
        VariablesTags,
        gsl::not_null<std::array<std::optional<StepperErrorEstimate>, 2>*>>::
        type... errors,
    const TimeStepper& time_stepper, const TimeStepId& time_step_id,
    const TimeStepId& next_time_step_id, const TimeDelta& time_step,
    const bool error_estimates_enabled,
    const TimeSteppers::History<typename VariablesTags::type>&... histories,
    const typename tmpl::has_type<
        VariablesTags, StepperErrorTolerances>::type&... tolerances) {
  if (::SelfStart::step_unused(time_step_id, next_time_step_id)) {
    return;
  }

  if (error_estimates_enabled) {
    const auto update_one_variables =
        [&time_stepper, &time_step]<typename Vars>(
            const gsl::not_null<Vars*> var,
            const gsl::not_null<
                std::array<std::optional<StepperErrorEstimate>, 2>*>
                error,
            const TimeSteppers::History<Vars>& history,
            StepperErrorTolerances tolerance) {
          // If we are doing variable-order h-refinement we need
          // low-order error estimates to restart elements.
          // Figuring out whether we are actually doing that is
          // hard, but since it can only happen at LTS slab
          // boundaries it's not a big deal if we do a little extra
          // work.  As of this writing, we do not use variable-order
          // steppers in GTS.
          if (tolerance.estimates != StepperErrorTolerances::Estimates::None and
              variants::holds_alternative<TimeSteppers::Tags::VariableOrder>(
                  time_stepper.order()) and
              (history.back().time_step_id.step_time() + time_step)
                  .is_at_slab_boundary()) {
            tolerance.estimates = StepperErrorTolerances::Estimates::AllOrders;
          }
          const auto new_error =
              time_stepper.update_u(var, history, time_step, tolerance);
          if (new_error.has_value()) {
            // Save the previous error if there was one, but not if this is a
            // retry of the same step.
            if ((*error)[1].has_value() and
                (*error)[1]->step_time != new_error->step_time) {
              (*error)[0] = (*error)[1];
            }
            (*error)[1].emplace(*new_error);
          }
          return 0;
        };
    expand_pack(update_one_variables(vars, errors, histories, tolerances)...);
  } else {
    const auto update_one_variables =
        [&time_stepper, &time_step]<typename Vars>(
            const gsl::not_null<Vars*> var,
            const TimeSteppers::History<Vars>& history) {
          time_stepper.update_u(var, history, time_step);
          return 0;
        };
    expand_pack(update_one_variables(vars, histories)...);
  }
}
