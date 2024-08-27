// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepperErrorEstimate.hpp"

#include <pup.h>

void StepperErrorEstimate::pup(PUP::er& p) {
  p | step_time;
  p | step_size;
  p | order;
  p | error;
}
