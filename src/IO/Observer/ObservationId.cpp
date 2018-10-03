// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ObservationId.hpp"

#include <ostream>
#include <pup.h>

namespace observers {
void ObservationId::pup(PUP::er& p) noexcept {
  p | combined_hash_;
  p | value_;
}

bool operator==(const ObservationId& lhs, const ObservationId& rhs) noexcept {
  return lhs.hash() == rhs.hash() and lhs.value() == rhs.value();
}

bool operator!=(const ObservationId& lhs, const ObservationId& rhs) noexcept {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ObservationId& t) noexcept {
  return os << '(' << t.hash() << ',' << t.value() << ')';
}
}  // namespace observers
