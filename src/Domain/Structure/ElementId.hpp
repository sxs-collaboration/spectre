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

#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
/// \endcond
namespace PUP {
class er;
}  // namespace PUP

/*!
 * \ingroup ComputationalDomainGroup
 * \brief An ElementId uniquely labels an Element.
 *
 * It is constructed from the BlockId of the Block to which the Element belongs
 * and the VolumeDim SegmentIds that label the segments of the Block that the
 * Element covers.
 *
 * \details
 * The `ElementId` serves as an index that is compatible with Charm++ and
 * therefore must adhere to the restrictions imposed by Charm++. These are:
 * - `ElementId` must satisfy `std::is_pod`
 * - `ElementId` must not be larger than the size of three `int`s, i.e.
 *   `sizeof(ElementId) <= 3 * sizeof(int)`
 *
 * The latter restriction can be relaxed with a special compilation flag to
 * Charm++, but we have not yet needed more elements than can be accounted for
 * by densely packing bits together. `SegmentId` is responsible for handling the
 * low-level bit manipulations to create an index that satisfies the size
 * constraints.
 */
template <size_t VolumeDim>
class ElementId {
 public:
  static constexpr size_t volume_dim = VolumeDim;

  /// Default constructor needed for Charm++ serialization.
  ElementId() noexcept = default;
  ElementId(const ElementId&) noexcept = default;
  ElementId& operator=(const ElementId&) noexcept = default;
  ElementId(ElementId&&) noexcept = default;
  ElementId& operator=(ElementId&&) noexcept = default;
  ~ElementId() noexcept = default;

  /// Create the ElementId of the root Element of a Block.
  explicit ElementId(size_t block_id) noexcept;

  /// Create an arbitrary ElementId.
  ElementId(size_t block_id,
            std::array<SegmentId, VolumeDim> segment_ids) noexcept;

  ElementId<VolumeDim> id_of_child(size_t dim, Side side) const noexcept;

  ElementId<VolumeDim> id_of_parent(size_t dim) const noexcept;

  size_t block_id() const noexcept {
    ASSERT(
        alg::all_of(
            segment_ids_,
            [this](const SegmentId& current_id) noexcept {
              return current_id.block_id() == segment_ids_[0].block_id();
            }),
        "Not all of the `SegmentId`s inside `ElementId` have same `BlockId`.");
    return segment_ids_[0].block_id();
  }

  const std::array<SegmentId, VolumeDim>& segment_ids() const noexcept {
    return segment_ids_;
  }

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

  /// Returns an ElementId meant for identifying data on external boundaries,
  /// which should never correspond to the Id of an actual element.
  static ElementId<VolumeDim> external_boundary_id() noexcept;

 private:
  std::array<SegmentId, VolumeDim> segment_ids_;
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
size_t hash_value(const ElementId<VolumeDim>& id) noexcept;

// clang-tidy: do not modify namespace std
namespace std {  // NOLINT
template <size_t VolumeDim>
struct hash<ElementId<VolumeDim>> {
  size_t operator()(const ElementId<VolumeDim>& id) const noexcept;
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
