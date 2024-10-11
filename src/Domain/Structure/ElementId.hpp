// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ElementId.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

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
 */
template <size_t VolumeDim>
class alignas(int[2]) ElementId {  // NOLINT(modernize-avoid-c-arrays)
 public:
  // We restrict the ElementId size to 64 bits for easy hashing into
  // size_t. This still allows us to have over 9 quadrillion elements, which is
  // probably enough. 2^45 * 2^8 9 quadrillion elements per grid index, with
  // up to 16 grid indices.
  //
  // Note: C++ populates bits from right to left in order of the
  // variables. This gives us the direction_mask we use below.
  static constexpr size_t block_id_bits = 8;
  static constexpr size_t grid_index_bits = 4;
  static constexpr size_t direction_bits = 4;
  /// The maximum allowed refinement level
  static constexpr size_t max_refinement_level = 15;
  static constexpr uint64_t direction_shift =
      static_cast<uint64_t>(block_id_bits + grid_index_bits);
  static constexpr uint64_t direction_mask = static_cast<uint64_t>(0b1111)
                                             << direction_shift;
  static_assert(block_id_bits + 3 * (1 + max_refinement_level) +
                        grid_index_bits + direction_bits ==
                    static_cast<size_t>(2 * 8) * sizeof(int),
                "Bit representation requires padding or is too large");

  static constexpr size_t volume_dim = VolumeDim;

  /// Default constructor needed for Charm++ serialization.
  ElementId() = default;
  ElementId(const ElementId&) = default;
  ElementId& operator=(const ElementId&) = default;
  ElementId(ElementId&&) = default;
  ElementId& operator=(ElementId&&) = default;
  ~ElementId() = default;

  /// Create the ElementId of the root Element of a Block.
  explicit ElementId(size_t block_id, size_t grid_index = 0);

  /// Create an arbitrary ElementId.
  ElementId(size_t block_id,
            const std::array<SegmentId, VolumeDim>& segment_ids,
            size_t grid_index = 0);

  /// Create an ElementId from its string representation (see `operator<<`).
  explicit ElementId(const std::string& grid_name);

  ElementId<VolumeDim> id_of_child(size_t dim, Side side) const;

  ElementId<VolumeDim> id_of_parent(size_t dim) const;

  size_t block_id() const { return block_id_; }

  size_t grid_index() const { return grid_index_; }

  std::array<size_t, VolumeDim> refinement_levels() const;

  std::array<SegmentId, VolumeDim> segment_ids() const;

  SegmentId segment_id(size_t dim) const;

  /// Returns an ElementId meant for identifying data on external boundaries,
  /// which does not correspond to the Id of an actual element.
  static ElementId<VolumeDim> external_boundary_id();

  /// Returns the number of block boundaries the element has.
  size_t number_of_block_boundaries() const;

 protected:
  /// Create an `ElementId` in a specified direction.
  ElementId(const Direction<VolumeDim>& direction,
            const ElementId<VolumeDim>& element_id);

  Direction<VolumeDim> direction() const;

  ElementId without_direction() const;

 private:
  template <size_t Dim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const ElementId<Dim>& lhs, const ElementId<Dim>& rhs);

  template <size_t Dim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator<(const ElementId<Dim>& lhs, const ElementId<Dim>& rhs);

  template <size_t Dim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool is_zeroth_element(const ElementId<Dim>& id,
                                const std::optional<size_t>& grid_index);

  ElementId(uint8_t block_id, uint8_t grid_index, uint8_t direction,
            uint16_t compact_segment_id_xi, uint16_t compact_segment_id_eta,
            uint16_t compact_segment_id_zeta);

  uint8_t block_id_ : block_id_bits;
  uint8_t grid_index_ : grid_index_bits;
  uint8_t direction_ : direction_bits;  // end first 16 bits
  // each of the following is 16 bits in length
  uint16_t compact_segment_id_xi_ : max_refinement_level + 1;
  uint16_t compact_segment_id_eta_ : max_refinement_level + 1;
  uint16_t compact_segment_id_zeta_ : max_refinement_level + 1;
};

/// \cond
// clang-format off
// macro that generate the pup operator for SegmentId
PUPbytes(ElementId<1>)      // NOLINT
PUPbytes(ElementId<2>)  // NOLINT
PUPbytes(ElementId<3>)  // NOLINT
/// \endcond

/// Output operator for ElementId.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const ElementId<VolumeDim>& id);
// clang-format on

/// Equivalence operator for ElementId.
template <size_t VolumeDim>
bool operator==(const ElementId<VolumeDim>& lhs,
                const ElementId<VolumeDim>& rhs);

/// Inequivalence operator for ElementId.
template <size_t VolumeDim>
bool operator!=(const ElementId<VolumeDim>& lhs,
                const ElementId<VolumeDim>& rhs);

/// Define an ordering of element IDs first by grid index, then by block ID,
/// then by segment ID in each dimension in turn. In each dimension, segment IDs
/// are ordered first by refinement level (which will typically be the same when
/// comparing two element IDs), and second by index. There's no particular
/// reason for this choice of ordering. For applications such as distributing
/// elements among cores, orderings such as defined by
/// `domain::BlockZCurveProcDistribution` may be more appropriate.
template <size_t VolumeDim>
bool operator<(const ElementId<VolumeDim>& lhs,
               const ElementId<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator>(const ElementId<VolumeDim>& lhs,
               const ElementId<VolumeDim>& rhs) {
  return rhs < lhs;
}
template <size_t VolumeDim>
bool operator<=(const ElementId<VolumeDim>& lhs,
                const ElementId<VolumeDim>& rhs) {
  return !(lhs > rhs);
}
template <size_t VolumeDim>
bool operator>=(const ElementId<VolumeDim>& lhs,
                const ElementId<VolumeDim>& rhs) {
  return !(lhs < rhs);
}

/// @{
/// \brief Returns a bool if the element is the zeroth element in the domain.
///
/// \details An element is considered to be the zeroth element if its ElementId
/// `id` has
/// 1. id.block_id() == 0
/// 2. All id.segment_ids() have SegmentId.index() == 0
/// 3. If the argument `grid_index` is specified, id.grid_index() == grid_index.
///
/// This means that the only element in a domain that this function will return
/// `true` for is the element in the lower corner of Block0 of that domain. The
/// `grid_index` will determine which domain is used for the comparison. During
/// evolutions, only one domain will be active at a time so it doesn't make
/// sense to compare the `grid_index`. However, during an elliptic solve
/// when there are multiple grids, this `grid_index` is useful for specifying
/// only one element over all domains.
///
/// This function is useful if you need a unique element in the domain because
/// only one element in the whole domain can be the zeroth element.
///
/// \parblock
/// \warning If you have multiple grids and you don't specify the `grid_index`
/// argument, this function will return `true` for one element in every grid
/// and thus can't be used to determine a unique element in a simulation; only a
/// unique element in each grid.
/// \endparblock
/// \parblock
/// \warning If the domain is re-gridded, a different ElementId may represent
/// the zeroth element.
/// \endparblock
template <size_t Dim>
bool is_zeroth_element(const ElementId<Dim>& id,
                       const std::optional<size_t>& grid_index);

// This overload is added (instead of adding a default value for grid_index)
// in order to avoid adding DomainStructures as a dependency of Parallel
// by using a forward declaration in Parallel/DistributedObject.hpp
template <size_t Dim>
bool is_zeroth_element(const ElementId<Dim>& id);
/// @}
// ######################################################################
// INLINE DEFINITIONS
// ######################################################################

template <size_t VolumeDim>
size_t hash_value(const ElementId<VolumeDim>& id);

// NOLINTNEXTLINE(cert-dcl58-cpp)
namespace std {
template <size_t VolumeDim>
struct hash<ElementId<VolumeDim>> {
  size_t operator()(const ElementId<VolumeDim>& id) const;
};
}  // namespace std

template <size_t VolumeDim>
inline bool operator==(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) {
  // Note: Direction is intentionally skipped.
  return lhs.block_id_ == rhs.block_id_ and
         lhs.grid_index_ == rhs.grid_index_ and
         lhs.compact_segment_id_xi_ == rhs.compact_segment_id_xi_ and
         lhs.compact_segment_id_eta_ == rhs.compact_segment_id_eta_ and
         lhs.compact_segment_id_zeta_ == rhs.compact_segment_id_zeta_;
}

template <size_t VolumeDim>
inline bool operator!=(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) {
  return not(lhs == rhs);
}
