// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"

#include <cmath>

#include "ErrorHandling/Assert.hpp"
#include "Time/TimeId.hpp"

namespace TimeSteppers {

uint64_t RungeKutta3::number_of_substeps() const noexcept {
  return 3;
}

size_t RungeKutta3::number_of_past_steps() const noexcept {
  return 0;
}

double RungeKutta3::stable_step() const noexcept {
  // This is the condition for  y' = -k y  to go to zero.
  return 0.5 * (1. + cbrt(4. + sqrt(17.)) - 1. / cbrt(4. + sqrt(17.)));
}

TimeId RungeKutta3::next_time_id(const TimeId& current_id,
                                 const TimeDelta& time_step) const noexcept {
  switch (current_id.substep()) {
    case 0:
      ASSERT(current_id.time() == current_id.step_time(), "Wrong substep time");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 1, current_id.step_time() + time_step};
    case 1:
      ASSERT(current_id.time() == current_id.step_time() + time_step,
             "Wrong substep time");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time(), 2,
              current_id.step_time() + time_step / 2};
    case 2:
      ASSERT(current_id.time() == current_id.step_time() + time_step / 2,
             "Wrong substep time");
      return {current_id.time_runs_forward(), current_id.slab_number(),
              current_id.step_time() + time_step};
    default:
      ERROR("Bad substep value in RK3: " << current_id.substep());
  }
}

}  // namespace TimeSteppers

/// \cond
PUP::able::PUP_ID TimeSteppers::RungeKutta3::my_PUP_ID =  // NOLINT
    0;
/// \endcond
