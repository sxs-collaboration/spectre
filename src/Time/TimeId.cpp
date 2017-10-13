// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeId.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

void TimeId::pup(PUP::er& p) noexcept {
  p | slab_number;
  p | time;
  p | substep;
}

std::ostream& operator<<(std::ostream& s, const TimeId& id) noexcept {
  return s << id.slab_number << ':' << id.time << ':' << id.substep;
}

size_t hash_value(const TimeId& id) noexcept {
  size_t h = 0;
  boost::hash_combine(h, id.slab_number);
  boost::hash_combine(h, id.time);
  boost::hash_combine(h, id.substep);
  return h;
}

namespace std {
size_t hash<TimeId>::operator()(const TimeId& id) const noexcept {
  return boost::hash<TimeId>{}(id);
}
}  // namespace std
