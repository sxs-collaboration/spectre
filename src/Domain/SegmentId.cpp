// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/SegmentId.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

SegmentId::SegmentId(const size_t refinement_level, const size_t index) noexcept
    : refinement_level_(refinement_level), index_(index) {
  ASSERT(index < two_to_the(refinement_level),
         "index = " << index << ", refinement_level = " << refinement_level);
}

void SegmentId::pup(PUP::er& p) noexcept {
  p | refinement_level_;
  p | index_;
}

std::ostream& operator<<(std::ostream& os, const SegmentId& id) noexcept {
  os << 'L' << id.refinement_level() << 'I' << id.index();
  return os;
}

// LCOV_EXCL_START
size_t hash_value(const SegmentId& s) noexcept {
  size_t h = 0;
  boost::hash_combine(h, s.refinement_level());
  boost::hash_combine(h, s.index());
  return h;
}
// LCOV_EXCL_STOP
