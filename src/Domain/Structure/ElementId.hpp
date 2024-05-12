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
#include <optional>

#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"

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
  ElementId() = default;
  ElementId(const ElementId&) = default;
  ElementId& operator=(const ElementId&) = default;
  ElementId(ElementId&&) = default;
  ElementId& operator=(ElementId&&) = default;
  ~ElementId() = default;

  /// Create the ElementId of the root Element of a Block.
  explicit ElementId(size_t block_id, size_t grid_index = 0);

  /// Create an arbitrary ElementId.
  ElementId(size_t block_id, std::array<SegmentId, VolumeDim> segment_ids,
            size_t grid_index = 0);

  /// Create an ElementId from its string representation (see `operator<<`).
  ElementId(const std::string& grid_name);

  ElementId<VolumeDim> id_of_child(size_t dim, Side side) const;

  ElementId<VolumeDim> id_of_parent(size_t dim) const;

  size_t block_id() const { return block_id_; }

  size_t grid_index() const { return grid_index_; }

  std::array<size_t, VolumeDim> refinement_levels() const {
    if constexpr (VolumeDim == 1) {
      return {{refinement_level_xi_}};
    } else if constexpr (VolumeDim == 2) {
      return {{refinement_level_xi_, refinement_level_eta_}};
    } else if constexpr (VolumeDim == 3) {
      return {{refinement_level_xi_, refinement_level_eta_,
               refinement_level_zeta_}};
    }
  }

  std::array<SegmentId, VolumeDim> segment_ids() const {
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

  SegmentId segment_id(const size_t dim) const {
    ASSERT(dim < VolumeDim, "Dimension must be smaller than "
                                << VolumeDim << ", but is: " << dim);
    switch (dim) {
      case 0:
        return {refinement_level_xi_, index_xi_};
      case 1:
        return {refinement_level_eta_, index_eta_};
      case 2:
        return {refinement_level_zeta_, index_zeta_};
      default:
        ERROR("Invalid dimension: " << dim);
    }
  }

  /// Returns an ElementId meant for identifying data on external boundaries,
  /// which should never correspond to the Id of an actual element.
  static ElementId<VolumeDim> external_boundary_id();

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

// clang-tidy: do not modify namespace std
namespace std {  // NOLINT
template <size_t VolumeDim>
struct hash<ElementId<VolumeDim>> {
  size_t operator()(const ElementId<VolumeDim>& id) const;
};
}  // namespace std

template <size_t VolumeDim>
inline bool operator==(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) {
  return lhs.block_id() == rhs.block_id() and
         lhs.segment_ids() == rhs.segment_ids() and
         lhs.grid_index() == rhs.grid_index();
}

template <size_t VolumeDim>
inline bool operator!=(const ElementId<VolumeDim>& lhs,
                       const ElementId<VolumeDim>& rhs) {
  return not(lhs == rhs);
}
