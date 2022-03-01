// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/DormandPrince5.hpp"

#include <cmath>
#include <limits>

#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

size_t DormandPrince5::order() const { return 5; }

size_t DormandPrince5::error_estimate_order() const { return 4; }

uint64_t DormandPrince5::number_of_substeps() const { return 6; }

uint64_t DormandPrince5::number_of_substeps_for_error() const { return 7; }

size_t DormandPrince5::number_of_past_steps() const { return 0; }

// The growth function for DP5 is
//
//   g = mu^6 / 600 + \sum_{n=0}^5 mu^n / n!,
//
// where mu = lambda * dt. The equation dy/dt = -lambda * y evolves
// stably if |g| < 1. For lambda=-2, chosen so the stable_step() for
// RK1 (i.e. forward Euler) would be 1, DP5 has a stable step
// determined by inserting mu->-2 dt into the above equation. Finding the
// solutions with a numerical root find yields a stable step of about 1.653.
double DormandPrince5::stable_step() const { return 1.6532839463174733; }

TimeStepId DormandPrince5::next_time_id(const TimeStepId& current_id,
                                        const TimeDelta& time_step) const {
  const auto& step = current_id.substep();
  const auto& t0 = current_id.step_time();
  const auto& t = current_id.substep_time();
  if (step < 6) {
    if (step == 0) {
      ASSERT(t == t0, "In DP5 substep 0, the substep time ("
                          << t << ") should equal t0 (" << t0 << ")");
    } else {
      ASSERT(t == t0 + gsl::at(c_, step - 1) * time_step,
             "In DP5 substep " << step << ", the substep time (" << t
                               << ") should equal t0+c[" << step - 1 << "]*dt ("
                               << t0 + gsl::at(c_, step - 1) * time_step
                               << ")");
    }
    if (step < 5) {
      return {current_id.time_runs_forward(), current_id.slab_number(), t0,
              step + 1, t0 + gsl::at(c_, step) * time_step};
    } else {
      return {current_id.time_runs_forward(), current_id.slab_number(),
              t0 + time_step};
    }
  } else {
    ERROR("In DP5 substep should be one of 0,1,2,3,4,5, not "
          << current_id.substep());
  }
}

TimeStepId DormandPrince5::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  const auto& step = current_id.substep();
  if (step < 5) {
    return next_time_id(current_id, time_step);
  } else {
    const auto& t0 = current_id.step_time();
    const auto& t = current_id.substep_time();
    ASSERT(t == t0 + gsl::at(c_, step - 1) * time_step,
           "In adaptive DP5 substep "
               << step << ", the substep time (" << t << ") should equal t0+c["
               << step - 1 << "]*dt (" << t0 + gsl::at(c_, step - 1) * time_step
               << ")");
    switch(step) {
      case 5:
        return {current_id.time_runs_forward(), current_id.slab_number(), t0,
                step + 1, t0 + gsl::at(c_, step) * time_step};
        break;
      case 6:
        return {current_id.time_runs_forward(), current_id.slab_number(),
                t0 + time_step};
        break;
      default:
        ERROR("In adaptive DP5 substep should be one of 0,1,2,3,4,5,6, not "
              << current_id.substep());
    }
  }
}

template <typename T>
void DormandPrince5::update_u_impl(
    const gsl::not_null<T*> u, const gsl::not_null<UntypedHistory<T>*> history,
    const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 5,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  const double dt = time_step.value();

  const auto increment_u = [&u, &history, &dt](const auto& coeffs_last,
                                               const auto& coeffs_this) {
    static_assert(std::tuple_size_v<std::decay_t<decltype(coeffs_last)>> + 1 ==
                      std::tuple_size_v<std::decay_t<decltype(coeffs_this)>>,
                  "Unexpected coefficient vector sizes.");
    *u = *history->untyped_most_recent_value() +
         coeffs_this.back() * dt * *(history->end() - 1).derivative();
    for (size_t i = 0; i < coeffs_last.size(); ++i) {
      *u += (gsl::at(coeffs_this, i) - gsl::at(coeffs_last, i)) * dt *
            *(history->begin() + static_cast<int>(i)).derivative();
    }
  };

  if (substep == 0) {
    *u = *history->untyped_most_recent_value() +
         (a2_[0] * dt) * *history->begin().derivative();
  } else if (substep == 1) {
    increment_u(a2_, a3_);
  } else if (substep == 2) {
    increment_u(a3_, a4_);
  } else if (substep == 3) {
    increment_u(a4_, a5_);
  } else if (substep == 4) {
    increment_u(a5_, a6_);
  } else if (substep == 5) {
    increment_u(a6_, b_);
  } else {
    ERROR("Substep in DP5 should be one of 0,1,2,3,4,5, not " << substep);
  }
}

template <typename T>
bool DormandPrince5::update_u_impl(
    const gsl::not_null<T*> u, const gsl::not_null<T*> u_error,
    const gsl::not_null<UntypedHistory<T>*> history,
    const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 5,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  if (substep < 6) {
    update_u_impl(u, history, time_step);
  } else if (substep == 6) {
    // u is the same as for the previous substep.
    *u = *history->untyped_most_recent_value();

    const double dt = time_step.value();

    *u_error = -b_alt_.back() * dt * *(history->end() - 1).derivative();
    for (size_t i = 0; i < b_.size(); ++i) {
      *u_error -= (gsl::at(b_alt_, i) - gsl::at(b_, i)) * dt *
                  *(history->begin() + static_cast<int>(i)).derivative();
    }
  } else {
    ERROR("Substep in adaptive DP5 should be one of 0,1,2,3,4,5,6, not "
          << substep);
  }
  return substep == 6;
}

template <typename T>
bool DormandPrince5::dense_update_u_impl(const gsl::not_null<T*> u,
                                         const UntypedHistory<T>& history,
                                         const double time) const {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double t0 = history.front().value();
  const double t_end = history.back().value();
  if (time == t_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = *history.untyped_most_recent_value();
    return true;
  }
  const evolution_less<double> before{t_end > t0};
  if (history.size() == 1 or before(t_end, time)) {
    return false;
  }
  const double dt = t_end - t0;
  const double output_fraction = (time - t0) / dt;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << t0);
  ASSERT(output_fraction <= 1.0, "Requested time ("
                                     << time << ") not within step [" << t0
                                     << ", " << t0 + dt << "]");

  // The formula for dense output is given in Numerical Recipes Sec. 17.2.3.
  // This version is modified to eliminate all the values of the function
  // except the most recent.
  const auto common = [&output_fraction](const size_t n) {
    return square(output_fraction) * gsl::at(d_, n) -
           (1.0 + 2.0 * output_fraction) * gsl::at(b_, n);
  };
  *u = *history.untyped_most_recent_value() +
       dt * (1.0 - output_fraction) *
           ((1.0 - output_fraction) *
                ((common(0) + output_fraction) * *history.begin().derivative() +
                 common(2) * *(history.begin() + 2).derivative() +
                 common(3) * *(history.begin() + 3).derivative() +
                 common(4) * *(history.begin() + 4).derivative() +
                 common(5) * *(history.begin() + 5).derivative()) +
            square(output_fraction) * ((1.0 - output_fraction) * d_[6] - 1.0) *
                *(history.begin() + 6).derivative());
  return true;
}

template <typename T>
bool DormandPrince5::can_change_step_size_impl(
    const TimeStepId& time_id, const UntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

const std::array<Time::rational_t, 6> DormandPrince5::c_ = {
    {{1, 5}, {3, 10}, {4, 5}, {8, 9}, {1, 1}, {1, 1}}};

TIME_STEPPER_DEFINE_OVERLOADS(DormandPrince5)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::DormandPrince5::my_PUP_ID =  // NOLINT
    0;
