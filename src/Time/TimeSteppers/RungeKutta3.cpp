// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"

#include <cmath>

#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace TimeSteppers {

size_t RungeKutta3::order() const { return 3; }

size_t RungeKutta3::error_estimate_order() const { return 2; }

uint64_t RungeKutta3::number_of_substeps() const { return 3; }

uint64_t RungeKutta3::number_of_substeps_for_error() const { return 3; }

size_t RungeKutta3::number_of_past_steps() const { return 0; }

double RungeKutta3::stable_step() const {
  // This is the condition for  y' = -k y  to go to zero.
  return 0.5 * (1. + cbrt(4. + sqrt(17.)) - 1. / cbrt(4. + sqrt(17.)));
}

TimeStepId RungeKutta3::next_time_id(const TimeStepId& current_id,
                                     const TimeDelta& time_step) const {
  switch (current_id.substep()) {
    case 0:
      ASSERT(current_id.substep_time() == current_id.step_time(),
             "Wrong substep time");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 1, current_id.step_time() + time_step};
    case 1:
      ASSERT(current_id.substep_time() == current_id.step_time() + time_step,
             "Wrong substep time");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 2,
              current_id.step_time() + time_step / 2};
    case 2:
      ASSERT(
          current_id.substep_time() == current_id.step_time() + time_step / 2,
          "Wrong substep time");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time() + time_step};
    default:
      ERROR("Bad substep value in RK3: " << current_id.substep());
  }
}

TimeStepId RungeKutta3::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

template <typename T>
void RungeKutta3::update_u_impl(const gsl::not_null<T*> u,
                                const gsl::not_null<UntypedHistory<T>*> history,
                                const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 3,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  switch (substep) {
    case 0: {
      // from (5.32) of Hesthaven
      // v^(1) = u^n + dt*RHS(u^n,t^n)
      *u = *history->untyped_most_recent_value() +
           time_step.value() * *history->begin().derivative();
      break;
    }
    case 1: {
      // from (5.32) of Hesthaven
      // v^(2) = (1/4)*( 3*u^n + v^(1) + dt*RHS(v^(1),t^n + dt) )
      *u = *history->untyped_most_recent_value() -
           0.25 * time_step.value() *
               (3.0 * *history->begin().derivative() -
                *(history->begin() + 1).derivative());
      break;
    }
    case 2: {
      // from (5.32) of Hesthaven
      // u^(n+1) = (1/3)*( u^n + 2*v^(2) + 2*dt*RHS(v^(2),t^n + (1/2)*dt) )
      *u = *history->untyped_most_recent_value() -
           (1.0 / 12.0) * time_step.value() *
               (*history->begin().derivative() +
                *(history->begin() + 1).derivative() -
                8.0 * *(history->begin() + 2).derivative());
      break;
    }
    default:
      ERROR("Bad substep value in RK3: " << substep);
  }
}

template <typename T>
bool RungeKutta3::update_u_impl(const gsl::not_null<T*> u,
                                const gsl::not_null<T*> u_error,
                                const gsl::not_null<UntypedHistory<T>*> history,
                                const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 3,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  update_u_impl(u, history, time_step);
  // error estimate is only available when completing a full step
  if ((history->end() - 1).time_step_id().substep() == 2) {
    // error is estimated by comparing the order 3 step result with an order 2
    // estimate. See e.g. Chapter II.4 of Harrier, Norsett, and Wagner 1993
    *u_error =
        -(1.0 / 3.0) * time_step.value() *
        (*history->begin().derivative() + *(history->begin() + 1).derivative() -
         2.0 * *(history->begin() + 2).derivative());
    return true;
  }
  return false;
}

template <typename T>
bool RungeKutta3::dense_update_u_impl(gsl::not_null<T*> u,
                                      const UntypedHistory<T>& history,
                                      const double time) const {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double step_start = history.front().value();
  const double step_end = history.back().value();
  if (time == step_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = *history.untyped_most_recent_value();
    return true;
  }
  const evolution_less<double> before{step_end > step_start};
  if (history.size() == 1 or before(step_end, time)) {
    return false;
  }
  const double time_step = step_end - step_start;
  const double output_fraction = (time - step_start) / time_step;
  ASSERT(output_fraction >= 0, "Attempting dense output at time " << time
         << ", but already progressed past " << step_start);
  ASSERT(output_fraction <= 1,
         "Requested time (" << time << " not within step [" << step_start
         << ", " << step_end << "]");

  // arXiv:1605.02429
  *u = *history.untyped_most_recent_value() -
       (1.0 / 6.0) * time_step * (1.0 - output_fraction) *
           ((1.0 - 5.0 * output_fraction) * *history.begin().derivative() +
            (1.0 + output_fraction) *
                (*(history.begin() + 1).derivative() +
                 4.0 * *(history.begin() + 2).derivative()));
  return true;
}

template <typename T>
bool RungeKutta3::can_change_step_size_impl(
    const TimeStepId& time_id, const UntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

TIME_STEPPER_DEFINE_OVERLOADS(RungeKutta3)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::RungeKutta3::my_PUP_ID =  // NOLINT
    0;
