// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/TimeStepper.hpp"

namespace TimeSteppers {
bool operator==(const VariableOrder& a, const VariableOrder& b) {
  return a.minimum == b.minimum && a.maximum == b.maximum;
}

bool operator!=(const VariableOrder& a, const VariableOrder& b) {
  return not(a == b);
}
}  // namespace TimeSteppers
