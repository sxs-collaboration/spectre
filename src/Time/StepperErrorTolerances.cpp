// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepperErrorTolerances.hpp"

#include <pup.h>

void StepperErrorTolerances::pup(PUP::er& p) {
  p | absolute;
  p | relative;
}

bool operator==(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b) {
  return a.absolute == b.absolute and a.relative == b.relative;
}

bool operator!=(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b) {
  return not(a == b);
}
