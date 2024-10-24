// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/SegmentId.hpp"

#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <functional>
#include <ostream>
#include <pup.h>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

SegmentId::SegmentId(const size_t refinement_level, const size_t index)
    : refinement_level_(refinement_level), index_(index) {
  ASSERT(index < two_to_the(refinement_level),
         "index = " << index << ", refinement_level = " << refinement_level);
}

void SegmentId::pup(PUP::er& p) {
  p | refinement_level_;
  p | index_;
}

std::ostream& operator<<(std::ostream& os, const SegmentId& id) {
  os << 'L' << id.refinement_level() << 'I' << id.index();
  return os;
}

// LCOV_EXCL_START
size_t hash_value(const SegmentId& segment_id) {
  size_t hash = 0;
  boost::hash_combine(hash, segment_id.refinement_level());
  boost::hash_combine(hash, segment_id.index());
  return hash;
}
// LCOV_EXCL_STOP

// NOLINTNEXTLINE(cert-dcl58-cpp)
namespace std {
size_t hash<SegmentId>::operator()(const SegmentId& segment_id) const {
  return hash_value(segment_id);
}
}  // namespace std
