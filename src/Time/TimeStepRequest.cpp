// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeStepRequest.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

void TimeStepRequest::pup(PUP::er& p) {
  p | size_goal;
  p | size;
  p | end;
  p | size_hard_limit;
  p | end_hard_limit;
}

bool operator==(const TimeStepRequest& a, const TimeStepRequest& b) {
  return a.size_goal == b.size_goal and a.size == b.size and a.end == b.end and
         a.size_hard_limit == b.size_hard_limit and
         a.end_hard_limit == b.end_hard_limit;
}

bool operator!=(const TimeStepRequest& a, const TimeStepRequest& b) {
  return not(a == b);
}
