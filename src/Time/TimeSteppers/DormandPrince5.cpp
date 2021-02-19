// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/DormandPrince5.hpp"

#include <cmath>
#include <limits>

#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

size_t DormandPrince5::order() const noexcept { return 5; }

size_t DormandPrince5::error_estimate_order() const noexcept { return 4; }

uint64_t DormandPrince5::number_of_substeps() const noexcept { return 6; }

uint64_t DormandPrince5::number_of_substeps_for_error() const noexcept {
  return 7;
}

size_t DormandPrince5::number_of_past_steps() const noexcept { return 0; }

// The growth function for DP5 is
//
//   g = mu^6 / 600 + \sum_{n=0}^5 mu^n / n!,
//
// where mu = lambda * dt. The equation dy/dt = -lambda * y evolves
// stably if |g| < 1. For lambda=-2, chosen so the stable_step() for
// RK1 (i.e. forward Euler) would be 1, DP5 has a stable step
// determined by inserting mu->-2 dt into the above equation. Finding the
// solutions with a numerical root find yields a stable step of about 1.653.
double DormandPrince5::stable_step() const noexcept {
  return 1.6532839463174733;
}

TimeStepId DormandPrince5::next_time_id(const TimeStepId& current_id,
                                        const TimeDelta& time_step) const
    noexcept {
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
    const TimeStepId& current_id, const TimeDelta& time_step) const noexcept {
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

const std::array<Time::rational_t, 6> DormandPrince5::c_ = {
    {{1, 5}, {3, 10}, {4, 5}, {8, 9}, {1, 1}, {1, 1}}};
}  // namespace TimeSteppers

/// \cond
PUP::able::PUP_ID TimeSteppers::DormandPrince5::my_PUP_ID =  // NOLINT
    0;
/// \endcond
