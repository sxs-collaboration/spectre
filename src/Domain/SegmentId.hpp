// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class SegmentId.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>

#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace PUP {
class er;
}  // namespace PUP

/*!
 *  \ingroup ComputationalDomainGroup
 *  \brief A SegmentId labels a segment of the interval [-1,1] and is used to
 *  identify the bounds of an Element in a Block in each dimension.
 *
 *  In \f$d\f$ dimensions, \f$d\f$ SegmentId%s are used to identify an Element.
 * In each dimension, a segment spans the subinterval \f$[-1 + 2 \frac{i}{N}, -1
 * + 2 \frac{i+1}{N}]\f$ of the logical coordinates of a Block, where \f$i \f$=
 * `index` and \f$N = 2^L\f$ where \f$L\f$ = `refinement_level`.
 *
 *  \image html SegmentId.svg  "SegmentIds"
 *
 * In the figure, The `index` of segments increase from the `lower side` to the
 * `upper side` in each dimension of a Block, while the `refinement level`
 * increases as the segments are subdivided.  For example, let the segment
 * labeled `self` be on `refinement level` \f$L\f$, with `index` \f$i\f$.  Its
 * `parent` segment is on `refinement level` \f$L-1\f$ with  `index`
 * \f$\frac{i-1}{2}\f$. The `children` of `self` are on `refinement level`
 * \f$L+1\f$, and have `index` \f$2i\f$ and \f$2i+1\f$ for the lower and upper
 * child respectively.  Also labeled on the figure are the `sibling` and
 * `abutting nibling` (child of sibling) of `self`. These relationships between
 * segments are important for h-refinement, since in each dimension an Element
 * can be flagged to split into its two `children` segments, or join with its
 * `sibling` segment to form its `parent` segment.  As refinement levels of
 * neighboring elements are kept within one, in the direction of its `sibling`,
 * a segment can only abut its `sibling` or `abutting nibling`, while on the
 * opposite side, it can abut a segment on its level, the next-lower, or the
 * next-higher level.
 */
class SegmentId {
 public:
  /// Default constructor needed for Charm++ serialization.
  constexpr SegmentId() noexcept = default;
  constexpr SegmentId(const SegmentId& segment_id) noexcept = default;
  constexpr SegmentId(SegmentId&& segment_id) noexcept = default;
  ~SegmentId() noexcept = default;
  SegmentId& operator=(const SegmentId& segment_id) noexcept = default;
  SegmentId& operator=(SegmentId&& segment_id) noexcept = default;

  SegmentId(size_t refinement_level, size_t index) noexcept;

  constexpr size_t refinement_level() const noexcept {
    return refinement_level_;
  }

  constexpr size_t index() const noexcept { return index_; }

  SegmentId id_of_parent() const noexcept;

  SegmentId id_of_child(Side side) const noexcept;

  /// The other child of the parent of this segment
  SegmentId id_of_sibling() const noexcept;

  /// The child of the sibling of this segment that shares an endpoint with it
  SegmentId id_of_abutting_nibling() const noexcept;

  /// The side on which this segment shares an endpoint with its sibling
  Side side_of_sibling() const noexcept;

  /// The id this segment would have if the coordinate axis were flipped.
  SegmentId id_if_flipped() const noexcept;

  /// The logical coordinate of the endpoint of the segment on the given Side.
  double endpoint(Side side) const noexcept;

  /// The logical coordinate of the midpoint of the segment
  double midpoint() const noexcept {
    return -1.0 + (1.0 + 2.0 * index_) / two_to_the(refinement_level_);
  }

  /// Does the segment overlap with another?
  bool overlaps(const SegmentId& other) const noexcept;

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  size_t refinement_level_ = std::numeric_limits<size_t>::max();
  size_t index_ = std::numeric_limits<size_t>::max();
};

/// Output operator for SegmentId.
std::ostream& operator<<(std::ostream& os, const SegmentId& id) noexcept;

/// Equivalence operator for SegmentId.
bool operator==(const SegmentId& lhs, const SegmentId& rhs) noexcept;

/// Inequivalence operator for SegmentId.
bool operator!=(const SegmentId& lhs, const SegmentId& rhs) noexcept;

//##############################################################################
// INLINE DEFINITIONS
//##############################################################################

inline SegmentId SegmentId::id_of_parent() const noexcept {
  ASSERT(0 != refinement_level_,
         "Cannot call id_of_parent() on root refinement level!");
  // The parent has half as many segments as the child.
  return {refinement_level_ - 1, index_ / 2};
}

inline SegmentId SegmentId::id_of_child(Side side) const noexcept {
  // We cannot ASSERT on the maximum level because it's only known at runtime
  // and only known elsewhere in the code, not by SegmentId. I.e. SegmentId is
  // too low-level to know about this.
  // The child has twice as many segments as the parent, so for a particular
  // parent segment, there is both an upper and lower child segment.
  if (Side::Lower == side) {
    return {refinement_level_ + 1, index_ * 2};
  }
  return {refinement_level_ + 1, 1 + index_ * 2};
}

inline SegmentId SegmentId::id_of_sibling() const noexcept {
  ASSERT(0 != refinement_level_,
         "The segment on the root refinement level has no sibling");
  return {refinement_level_, (0 == index_ % 2 ? index_ + 1 : index_ - 1)};
}

inline SegmentId SegmentId::id_of_abutting_nibling() const noexcept {
  ASSERT(0 != refinement_level_,
         "The segment on the root refinement level has no abutting nibling");
  return {refinement_level_ + 1,
          (0 == index_ % 2 ? 2 * index_ + 2 : 2 * index_ - 1)};
}

inline Side SegmentId::side_of_sibling() const noexcept {
  ASSERT(0 != refinement_level_,
         "The segment on the root refinement level has no sibling");
  return 0 == index_ % 2 ? Side::Upper : Side::Lower;
}

inline SegmentId SegmentId::id_if_flipped() const noexcept {
  return {refinement_level_, two_to_the(refinement_level_) - 1 - index_};
}

inline double SegmentId::endpoint(Side side) const noexcept {
  if (Side::Lower == side) {
    return -1.0 + (2.0 * index_) / two_to_the(refinement_level_);
  }
  return -1.0 + (2.0 * index_ + 2.0) / two_to_the(refinement_level_);
}

inline bool SegmentId::overlaps(const SegmentId& other) const noexcept {
  const size_t this_denom = two_to_the(refinement_level_);
  const size_t other_denom = two_to_the(other.refinement_level_);
  return index_ * other_denom < (other.index_ + 1) * this_denom and
         other.index_ * this_denom < (index_ + 1) * other_denom;
}

// These are defined so that a SegmentId can be used as part of a key of an
// unordered_set or unordered_map.

// hash_value is called by boost::hash and related functions.
size_t hash_value(const SegmentId& s) noexcept;

inline bool operator==(const SegmentId& lhs, const SegmentId& rhs) noexcept {
  return (lhs.refinement_level() == rhs.refinement_level() and
          lhs.index() == rhs.index());
}

inline bool operator!=(const SegmentId& lhs, const SegmentId& rhs) noexcept {
  return not(lhs == rhs);
}
