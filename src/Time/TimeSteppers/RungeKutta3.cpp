// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"

#include <cmath>

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

}  // namespace TimeSteppers
