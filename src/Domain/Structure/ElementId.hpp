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
 * Element covers. An optional `grid_index` identifies elements with the same
 * BlockId and SegmentIds across multiple grids.
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
  static constexpr size_t block_id_bits = 24;
  static constexpr size_t grid_index_bits = 8;
  static constexpr size_t refinement_bits = 4;
  static constexpr size_t max_refinement_level = 16;
  // We need some padding to ensure bit fields align with type boundaries,
  // otherwise the size of `ElementId` is too large.
  static constexpr size_t padding = 4;
  static_assert(block_id_bits + 3 * (refinement_bits + max_refinement_level) +
                        grid_index_bits + padding ==
                    3 * 8 * sizeof(int),
                "Bit representation requires padding or is too large");
  static_assert(two_to_the(refinement_bits) >= max_refinement_level,
                "Not enough bits to represent all refinement levels");

  static constexpr size_t volume_dim = VolumeDim;

  /// Default constructor needed for Charm++ serialization.
  ElementId() noexcept = default;
  ElementId(const ElementId&) noexcept = default;
  ElementId& operator=(const ElementId&) noexcept = default;
  ElementId(ElementId&&) noexcept = default;
  ElementId& operator=(ElementId&&) noexcept = default;
  ~ElementId() noexcept = default;

  /// Create the ElementId of the root Element of a Block.
  explicit ElementId(size_t block_id, size_t grid_index = 0) noexcept;

  /// Create an arbitrary ElementId.
  ElementId(size_t block_id, std::array<SegmentId, VolumeDim> segment_ids,
            size_t grid_index = 0) noexcept;

  ElementId<VolumeDim> id_of_child(size_t dim, Side side) const noexcept;

  ElementId<VolumeDim> id_of_parent(size_t dim) const noexcept;

  size_t block_id() const noexcept { return block_id_; }

  size_t grid_index() const noexcept { return grid_index_; }

  std::array<SegmentId, VolumeDim> segment_ids() const noexcept {
    if constexpr (VolumeDim == 1) {
      return {{SegmentId{refinement_level_xi_, index_xi_}}};
    } else if constexpr (VolumeDim == 2) {
      return {{SegmentId{refinement_level_xi_, index_xi_},
               SegmentId{refinement_level_eta_, index_eta_}}};
    } else if constexpr (VolumeDim == 3) {
      return {{SegmentId{refinement_level_xi_, index_xi_},
               SegmentId{refinement_level_eta_, index_eta_},
               SegmentId{refinement_level_zeta_, index_zeta_}}};
    }
  }

  /// Returns an ElementId meant for identifying data on external boundaries,
  /// which should never correspond to the Id of an actual element.
  static ElementId<VolumeDim> external_boundary_id() noexcept;

 private:
  uint32_t block_id_ : block_id_bits;
  uint32_t grid_index_ : grid_index_bits;  // end first 32 bits
  uint32_t index_xi_ : max_refinement_level;
  uint32_t index_eta_ : max_refinement_level;
  uint32_t index_zeta_ : max_refinement_level;
  uint32_t empty_ : padding;
  uint32_t refinement_level_xi_ : refinement_bits;  // end second 32 bits
  uint32_t refinement_level_eta_ : refinement_bits;
  uint32_t refinement_level_zeta_ : refinement_bits;  // end third 32 bits
};

/// \cond
// macro that generate the pup operator for SegmentId
PUPbytes(ElementId<1>)      // NOLINT
PUPbytes(ElementId<2>)  // NOLINT
PUPbytes(ElementId<3>)  // NOLINT
/// \endcond

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
         lhs.segment_ids() == rhs.segment_ids() and
         lhs.grid_index() == rhs.grid_index();
}

template <size_t VolumeDim>
inline bool operator!=(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}
