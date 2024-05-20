// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/SelfStart.hpp"

#include "Time/TimeStepId.hpp"

namespace SelfStart {
bool is_self_starting(const TimeStepId& time_id) {
  return time_id.slab_number() < 0;
}

bool step_unused(const TimeStepId& time_id, const TimeStepId& next_time_id) {
  return time_id.slab_number() < 0 and
         time_id.slab_number() != next_time_id.slab_number();
}
}  // namespace SelfStart
