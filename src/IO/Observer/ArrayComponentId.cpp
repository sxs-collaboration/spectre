// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ArrayComponentId.hpp"

#include <pup.h>

namespace observers {
void ArrayComponentId::pup(PUP::er& p) {
  p | component_id_;
  p | array_index_;
}

bool operator==(const ArrayComponentId& lhs, const ArrayComponentId& rhs) {
  return lhs.component_id() == rhs.component_id() and
         lhs.array_index() == rhs.array_index();
}

bool operator!=(const ArrayComponentId& lhs, const ArrayComponentId& rhs) {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os,
                         const ArrayComponentId& array_component_id) {
  return os << std::hash<ArrayComponentId>{}(array_component_id);
}
}  // namespace observers
