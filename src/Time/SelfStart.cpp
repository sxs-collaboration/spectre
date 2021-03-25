// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/SelfStart.hpp"

#include "Time/TimeStepId.hpp"

namespace SelfStart {
bool is_self_starting(const TimeStepId& time_id) noexcept {
  return time_id.slab_number() < 0;
}
}  // namespace SelfStart
