// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ArrayComponentId.hpp"

#include <pup.h>

namespace observers {
void ArrayComponentId::pup(PUP::er& p) noexcept {
  p | component_id_;
  p | array_index_;
}

bool operator==(const ArrayComponentId& lhs,
                const ArrayComponentId& rhs) noexcept {
  return lhs.component_id() == rhs.component_id() and
         lhs.array_index() == rhs.array_index();
}

bool operator!=(const ArrayComponentId& lhs,
                const ArrayComponentId& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace observers
