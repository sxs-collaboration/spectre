// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta4.hpp"

#include <cmath>
#include <limits>

#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace TimeSteppers {

size_t RungeKutta4::order() const { return 4; }

size_t RungeKutta4::error_estimate_order() const { return 3; }

uint64_t RungeKutta4::number_of_substeps() const { return 4; }

uint64_t RungeKutta4::number_of_substeps_for_error() const { return 5; }

size_t RungeKutta4::number_of_past_steps() const { return 0; }

// The growth function for RK4 is (e.g. page 60 of
// http://www.staff.science.uu.nl/~frank011/Classes/numwisk/ch10.pdf
//
//   g = 1 + mu + mu^2 / 2 + mu^3 / 6 + mu^4 / 24,
//
// where mu = lambda * dt. The equation dy/dt = -lambda * y evolves
// stably if |g| < 1. For lambda=-2, chosen so the stable_step() for
// RK1 (i.e. forward Euler) would be 1, RK4 has a stable step
// determined by inserting mu->-2 dt into the above equation. Finding the
// solutions with a numerical root find yields a stable step of about 1.39265.
double RungeKutta4::stable_step() const { return 1.3926467817026411; }

TimeStepId RungeKutta4::next_time_id(const TimeStepId& current_id,
                                     const TimeDelta& time_step) const {
  switch (current_id.substep()) {
    case 0:
      ASSERT(current_id.substep_time() == current_id.step_time(),
             "In RK4 substep 0, the substep time ("
                 << current_id.substep_time() << ") should equal t0 ("
                 << current_id.step_time() << ")");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 1,
              current_id.step_time() + time_step / 2};
    case 1:
      ASSERT(
          current_id.substep_time() == current_id.step_time() + time_step / 2,
          "In RK4 substep 1, the substep time ("
              << current_id.substep_time() << ") should equal t0+0.5*dt ("
              << current_id.step_time() + time_step / 2 << ")");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 2,
              current_id.step_time() + time_step / 2};
    case 2:
      ASSERT(
          current_id.substep_time() == current_id.step_time() + time_step / 2,
          "In RK4 substep 2, the substep time ("
              << current_id.substep_time() << ") should equal t0+0.5*dt ("
              << current_id.step_time() + time_step / 2 << ")");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 3, current_id.step_time() + time_step};
    case 3:
      ASSERT(current_id.substep_time() == current_id.step_time() + time_step,
             "In RK4 substep 3, the substep time ("
                 << current_id.substep_time() << ") should equal t0+dt ("
                 << current_id.step_time() + time_step << ")");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time() + time_step};
    default:
      ERROR("In RK4 substep should be one of 0,1,2,3, not "
            << current_id.substep());
  }
}

TimeStepId RungeKutta4::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  if (current_id.substep() < 3) {
    return next_time_id(current_id, time_step);
  }
  // The embedded Zonneveld 4(3) scheme adds an extra substep at 3/4 that is
  // used only by the third-order error estimation scheme
  switch (current_id.substep()) {
    case 3:
      ASSERT(current_id.substep_time() == current_id.step_time() + time_step,
             "In adaptive RK4 substep 3, the substep time ("
                 << current_id.substep_time() << ") should equal t0+dt ("
                 << current_id.step_time() + time_step << ")");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 4,
              current_id.step_time() + 3 * time_step / 4};
    case 4:
      ASSERT(current_id.substep_time() ==
                 current_id.step_time() + 3 * time_step / 4,
             "In adaptive RK4 substep 4, the substep time ("
                 << current_id.substep_time() << ") should equal t0+dt ("
                 << current_id.step_time() + 3 * time_step / 4 << ")");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time() + time_step};
    default:
      ERROR("In adaptive RK4 substep should be one of 0,1,2,3,4 not "
            << current_id.substep());
  }
}

template <typename T>
void RungeKutta4::update_u_impl(const gsl::not_null<T*> u,
                                const gsl::not_null<UntypedHistory<T>*> history,
                                const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 4,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  switch (substep) {
    case 0: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(1) = u^n + dt * \mathcal{L}(u^n,t^n)/2
      *u = *history->untyped_most_recent_value() +
           0.5 * time_step.value() * *history->begin().derivative();
      break;
    }
    case 1: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(2) = u^n + dt * \mathcal{L}(v^(1), t^n + (1/2)*dt)/2
      *u = *history->untyped_most_recent_value() -
           0.5 * time_step.value() *
               (*history->begin().derivative() -
                *(history->begin() + 1).derivative());
      break;
    }
    case 2: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(3) = u^n + dt * \mathcal{L}(v^(2), t^n + (1/2)*dt))
      *u = *history->untyped_most_recent_value() +
           time_step.value() * (-0.5 * *(history->begin() + 1).derivative() +
                                *(history->begin() + 2).derivative());
      break;
    }
    case 3: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3) + v^(4) - 3*u0)/6
      // Note: v^(4) = u0 + dt * \mathcal{L}(t+dt, v^(3)); inserting this gives
      // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3)
      //         + dt*\mathcal{L}(t+dt,v^(3)) - 2*u0)/6
      *u = *history->untyped_most_recent_value() +
           (1.0 / 3.0) * time_step.value() *
               (0.5 * *history->begin().derivative() +
                *(history->begin() + 1).derivative() -
                2.0 * *(history->begin() + 2).derivative() +
                0.5 * *(history->begin() + 3).derivative());
      break;
    }
    default:
      ERROR("Substep in RK4 should be one of 0,1,2,3, not " << substep);
  }
}

template <typename T>
bool RungeKutta4::update_u_impl(const gsl::not_null<T*> u,
                                const gsl::not_null<T*> u_error,
                                const gsl::not_null<UntypedHistory<T>*> history,
                                const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 4,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();
  if (substep < 3) {
    update_u_impl(u, history, time_step);
  } else {
    switch (substep) {
      case 3: {
        *u = *history->untyped_most_recent_value() +
             (1.0 / 32.0) * time_step.value() *
                 (5.0 * *history->begin().derivative() +
                  7.0 * *(history->begin() + 1).derivative() -
                  19.0 * *(history->begin() + 2).derivative() -
                  *(history->begin() + 3).derivative());
        break;
      }
      case 4: {
        // from (17.1.3) of Numerical Recipes 3rd Edition
        // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3) + v^(4) - 3*u0)/6
        // Note: v^(4) = u0 + dt * \mathcal{L}(t+dt, v^(3)); inserting this
        // gives u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3)
        //         + dt*\mathcal{L}(t+dt,v^(3)) - 2*u0)/6
        *u = *history->untyped_most_recent_value() +
             (1.0 / 96.0) * time_step.value() *
                 (*history->begin().derivative() +
                  11.0 * *(history->begin() + 1).derivative() -
                  7.0 * *(history->begin() + 2).derivative() +
                  19.0 * *(history->begin() + 3).derivative());

        // See Butcher Tableau of Zonneveld 4(3) embedded scheme with five
        // substeps in Table 4.2 of Hairer, Norsett, and Wanner
        *u_error = (2.0 / 3.0) * time_step.value() *
                   (*history->begin().derivative() -
                    3.0 * (*(history->begin() + 1).derivative() +
                           *(history->begin() + 2).derivative() +
                           *(history->begin() + 3).derivative()) +
                    8.0 * *(history->begin() + 4).derivative());
        break;
      }
      default:
        ERROR("Substep in adaptive RK4 should be one of 0,1,2,3,4, not "
              << substep);
    }
  }
  return substep == 4;
}

template <typename T>
bool RungeKutta4::dense_update_u_impl(const gsl::not_null<T*> u,
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
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << step_start);
  ASSERT(output_fraction <= 1.0, "Requested time ("
                                     << time << ") not within step ["
                                     << step_start << ", " << step_end << "]");

  // Numerical Recipes Eq. (17.2.15). This implements cubic interpolation
  // throughout the step.
  *u = *history.untyped_most_recent_value() -
       (1.0 / 3.0) * time_step * (1.0 - output_fraction) *
           ((1.0 - output_fraction) *
                ((0.5 - 2.0 * output_fraction) * *history.begin().derivative() +
                 (1.0 + 2.0 * output_fraction) *
                     (*(history.begin() + 1).derivative() +
                      *(history.begin() + 2).derivative() +
                      0.5 * *(history.begin() + 3).derivative())) +
            3.0 * square(output_fraction) * *(history.end() - 1).derivative());
  return true;
}

template <typename T>
bool RungeKutta4::can_change_step_size_impl(
    const TimeStepId& time_id, const UntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

TIME_STEPPER_DEFINE_OVERLOADS(RungeKutta4)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::RungeKutta4::my_PUP_ID =  // NOLINT
    0;
