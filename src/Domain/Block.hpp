// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Block.

#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <unordered_set>

#include "Domain/BlockNeighbor.hpp"                 // IWYU pragma: keep
#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"                     // IWYU pragma: keep
#include "Domain/DirectionMap.hpp"

/// \cond
namespace Frame {
struct Logical;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// A Block<VolumeDim> is a region of a VolumeDim-dimensional computational
/// domain that defines the root node of a tree which is used to construct the
/// Elements that cover a region of the computational domain.
///
/// Each codimension 1 boundary of a Block<VolumeDim> is either an external
/// boundary or identical to a boundary of one other Block.
///
/// A Block has logical coordinates that go from -1 to +1 in each
/// dimension.  The global coordinates are obtained from the logical
/// coordinates from the Coordinatemap:  CoordinateMap::operator() takes
/// Points in the Logical Frame (i.e., logical coordinates) and
/// returns Points in the Grid Frame (i.e., global coordinates).
template <size_t VolumeDim, typename TargetFrame>
class Block {
 public:
  /// \param map the CoordinateMap.
  /// \param id a unique ID.
  /// \param neighbors info about the Blocks that share a codimension 1
  /// boundary with this Block.
  Block(std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame,
                                                  VolumeDim>>&& map,
        size_t id,
        DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors) noexcept;

  Block() = default;
  ~Block() = default;
  Block(const Block&) = delete;
  Block(Block&&) = default;
  Block& operator=(const Block&) = delete;
  Block& operator=(Block&&) = default;

  const domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>&
  coordinate_map() const noexcept {
    return *map_;
  }

  /// A unique identifier for the Block that is in the range
  /// [0, number_of_blocks -1] where number_of_blocks is the number
  /// of Blocks that cover the computational domain.
  size_t id() const noexcept { return id_; }

  /// Information about the neighboring Blocks.
  const DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>& neighbors() const
      noexcept {
    return neighbors_;
  }

  /// The directions of the faces of the Block that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const
      noexcept {
    return external_boundaries_;
  }

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>
      map_;
  size_t id_{0};
  DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors_;
  std::unordered_set<Direction<VolumeDim>> external_boundaries_;
};

template <size_t VolumeDim, typename TargetFrame>
std::ostream& operator<<(std::ostream& os,
                         const Block<VolumeDim, TargetFrame>& block) noexcept;

template <size_t VolumeDim, typename TargetFrame>
bool operator==(const Block<VolumeDim, TargetFrame>& lhs,
                const Block<VolumeDim, TargetFrame>& rhs) noexcept;

template <size_t VolumeDim, typename TargetFrame>
bool operator!=(const Block<VolumeDim, TargetFrame>& lhs,
                const Block<VolumeDim, TargetFrame>& rhs) noexcept;
