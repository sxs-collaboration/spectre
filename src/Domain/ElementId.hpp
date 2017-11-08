// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ElementId.

#pragma once

#include <array>
#include <iosfwd>
#include <pup.h>

#include "Domain/SegmentId.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t>
class ElementIndex;

/// \ingroup ComputationalDomainGroup
/// An ElementId uniquely labels a Element.
/// It is constructed from the BlockId of the Block to which the Element belongs
/// and the VolumeDim SegementIds that label the segments of the Block that the
/// Element covers.
template <size_t VolumeDim>
class ElementId {
 public:
  /// Default constructor needed for Charm++ serialization.
  constexpr ElementId() = default;

  /// Create the ElementId of the root Element of a Block.
  explicit ElementId(size_t block_id) noexcept;

  /// Convert an ElementIndex to an ElementId
  // clang-tidy: mark explicit: we want to allow conversion
  ElementId(const ElementIndex<VolumeDim>& index) noexcept;  // NOLINT

  /// Create an arbitrary ElementId.
  ElementId(size_t block_id,
            std::array<SegmentId, VolumeDim> segment_ids) noexcept;

  ElementId<VolumeDim> id_of_child(size_t dim, Side side) const noexcept;

  ElementId<VolumeDim> id_of_parent(size_t dim) const noexcept;

  constexpr size_t block_id() const noexcept { return block_id_; }

  const std::array<SegmentId, VolumeDim>& segment_ids() const noexcept {
    return segment_ids_;
  }

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  size_t block_id_ = std::numeric_limits<size_t>::max();
  std::array<SegmentId, VolumeDim> segment_ids_ =
      make_array<VolumeDim>(SegmentId());
};

/// Output operator for ElementId.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const ElementId<VolumeDim>& id);

/// Equivalence operator for ElementId.
template <size_t VolumeDim>
bool operator==(const ElementId<VolumeDim>& lhs,
                const ElementId<VolumeDim>& rhs) noexcept;

/// Inequivalence operator for ElementId.
template <size_t VolumeDim>
bool operator!=(const ElementId<VolumeDim>& lhs,
                const ElementId<VolumeDim>& rhs) noexcept;

// ######################################################################
// INLINE DEFINITIONS
// ######################################################################

template <size_t VolumeDim>
size_t hash_value(const ElementId<VolumeDim>& c) noexcept;

namespace std {
template <size_t VolumeDim>
struct hash<ElementId<VolumeDim>> {
  size_t operator()(const ElementId<VolumeDim>& c) const noexcept;
};
}  // namespace std

template <size_t VolumeDim>
inline bool operator==(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) noexcept {
  return lhs.block_id() == rhs.block_id() and
         lhs.segment_ids() == rhs.segment_ids();
}

template <size_t VolumeDim>
inline bool operator!=(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}
