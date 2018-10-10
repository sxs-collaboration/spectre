// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ElementId.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <limits>

#include "Domain/ElementIndex.hpp" // IWYU pragma: keep
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Parallel {
template <class>
class ArrayIndex;
}  // namespace Parallel
/// \endcond
namespace PUP {
class er;
}  // namespace PUP

/// \ingroup ComputationalDomainGroup
/// An ElementId uniquely labels an Element.
/// It is constructed from the BlockId of the Block to which the Element belongs
/// and the VolumeDim SegmentIds that label the segments of the Block that the
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

  /// Conversion operator needed to index `proxy`s using `ElementId`
  // clang-tidy: mark explicit, we want implicit conversion
  operator Parallel::ArrayIndex<ElementIndex<VolumeDim>>() const  // NOLINT
      noexcept;

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

  /// Returns an ElementId meant for identifying data on external boundaries,
  /// which should never correspond to the Id of an actual element.
  static ElementId<VolumeDim> external_boundary_id() noexcept;

 private:
  size_t block_id_ = std::numeric_limits<size_t>::max();
  std::array<SegmentId, VolumeDim> segment_ids_ =
      make_array<VolumeDim>(SegmentId());
};

/// Output operator for ElementId.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ElementId<VolumeDim>& id) noexcept;

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

// clang-tidy: do not modify namespace std
namespace std {  // NOLINT
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
