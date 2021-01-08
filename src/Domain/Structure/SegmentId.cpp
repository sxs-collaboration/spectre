// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/SegmentId.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

SegmentId::SegmentId(const size_t refinement_level, const size_t index) noexcept
    : block_id_(0), refinement_level_(refinement_level), index_(index) {
  ASSERT(refinement_level <= max_refinement_level,
         "Refinement level out of bounds: " << refinement_level);
  ASSERT(index < two_to_the(refinement_level),
         "index = " << index << ", refinement_level = " << refinement_level);
}

SegmentId::SegmentId(const size_t block_id, const size_t refinement_level,
                     const size_t index) noexcept
    : block_id_(block_id), refinement_level_(refinement_level), index_(index) {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(refinement_level <= max_refinement_level,
         "Refinement level out of bounds: " << refinement_level
                                            << "\nMaximum value is: "
                                            << max_refinement_level);
  ASSERT(index < two_to_the(refinement_level),
         "index = " << index << ", refinement_level = " << refinement_level);
}

// Because `ElementId` is built up of `VolumeDim` `SegmentId`s, and `ElementId`
// is used as a Charm++ index, `ElementId` has the following restrictions:
// - `ElementId` must satisfy `std::is_pod`
// - `ElementId` must not be larger than the size of three `int`s, i.e.
//   `sizeof(ElementId) <= 3 * sizeof(int)`
// which means `SegmentId` must be the size of an `int` and satisfy
// `std::is_pod`.
static_assert(std::is_pod<SegmentId>::value, "SegmentId is not POD");
static_assert(sizeof(SegmentId) == sizeof(int),
              "SegmentId does not fit in an int");

std::ostream& operator<<(std::ostream& os, const SegmentId& id) noexcept {
  os << 'L' << id.refinement_level() << 'I' << id.index();
  return os;
}

// LCOV_EXCL_START
size_t hash_value(const SegmentId& segment_id) noexcept {
  size_t hash = 0;
  boost::hash_combine(hash, segment_id.refinement_level());
  boost::hash_combine(hash, segment_id.index());
  return hash;
}
// LCOV_EXCL_STOP
