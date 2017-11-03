// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"

#include <cmath>

#include "Time/TimeId.hpp"

namespace TimeSteppers {

size_t RungeKutta3::number_of_substeps() const noexcept {
  return 3;
}

size_t RungeKutta3::number_of_past_steps() const noexcept {
  return 0;
}

bool RungeKutta3::is_self_starting() const noexcept {
  return true;
}

double RungeKutta3::stable_step() const noexcept {
  // This is the condition for  y' = -k y  to go to zero.
  return 0.5 * (1. + cbrt(4. + sqrt(17.)) - 1. / cbrt(4. + sqrt(17.)));
}

TimeId RungeKutta3::next_time_id(const TimeId& current_id,
                                 const TimeDelta& time_step) const noexcept {
  TimeId next_id = current_id;
  switch (current_id.substep) {
    case 0:
      next_id.time += time_step;
      next_id.substep = 1;
      return next_id;
    case 1:
      next_id.time -= time_step / 2;
      next_id.substep = 2;
      return next_id;
    case 2:
      next_id.time += time_step / 2;
      next_id.substep = 0;
      return next_id;
    default:
      ERROR("Bad substep value in RK3: " << current_id.substep);
  }
}

}  // namespace TimeSteppers

/// \cond
PUP::able::PUP_ID TimeSteppers::RungeKutta3::my_PUP_ID =  // NOLINT
    0;
/// \endcond
