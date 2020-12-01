// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta4.hpp"

#include <cmath>
#include <limits>

#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace TimeSteppers {

uint64_t RungeKutta4::number_of_substeps() const noexcept { return 4; }

uint64_t RungeKutta4::number_of_substeps_for_error() const noexcept {
  return 5;
}

size_t RungeKutta4::number_of_past_steps() const noexcept { return 0; }

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
double RungeKutta4::stable_step() const noexcept { return 1.3926467817026411; }

TimeStepId RungeKutta4::next_time_id(const TimeStepId& current_id,
                                     const TimeDelta& time_step) const
    noexcept {
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
    const TimeStepId& current_id, const TimeDelta& time_step) const noexcept {
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
}  // namespace TimeSteppers

/// \cond
PUP::able::PUP_ID TimeSteppers::RungeKutta4::my_PUP_ID =  // NOLINT
    0;
/// \endcond
