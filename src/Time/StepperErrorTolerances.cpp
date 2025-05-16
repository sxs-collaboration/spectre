// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepperErrorTolerances.hpp"

#include <pup.h>
#include <pup_stl.h>  // defines pup for enum

void StepperErrorTolerances::pup(PUP::er& p) {
  p | estimates;
  p | absolute;
  p | relative;
}

bool operator==(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b) {
  return (a.estimates == StepperErrorTolerances::Estimates::None and
          b.estimates == StepperErrorTolerances::Estimates::None) or
         (a.estimates == b.estimates and a.absolute == b.absolute and
          a.relative == b.relative);
}

bool operator!=(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b) {
  return not(a == b);
}
