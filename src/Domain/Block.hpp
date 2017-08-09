// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Block.

#pragma once

#include <iosfwd>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "Domain/BlockNeighbor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/EmbeddingMaps/EmbeddingMap.hpp"

/// \ingroup ComputationalDomain
/// A Block<VolumeDim> is a region of a VolumeDim-dimensional computational
/// domain that defines the root node of a tree which is used to construct the
/// Elements that cover a region of the computational domain.
///
/// Each codimension 1 boundary of a Block<VolumeDim> is either an external
/// boundary or identical to a boundary of one other Block.
///
/// A Block has logical coordinates that go from -1 to +1 in each
/// dimension.  The global coordinates are obtained from the logical
/// coordinates from the EmbeddingMap:  EmbeddingMap::operator() takes
/// Points in the Logical Frame (i.e., logical coordinates) and
/// returns Points in the Grid Frame (i.e., global coordinates).
///
/// \tparam VolumeDim the dimension of the Block (i.e. the volume dimension).
template <size_t VolumeDim>
class Block {
 public:
  /// \param embedding_map the EmbeddingMap.
  /// \param id a unique ID.
  /// \param neighbors info about the Blocks that share a codimension 1
  /// boundary with this Block.
  Block(const EmbeddingMap<VolumeDim, VolumeDim>& embedding_map, size_t id,
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>&&
            neighbors);

  Block() = default;
  ~Block() = default;
  Block(const Block<VolumeDim>&) = delete;
  Block(Block<VolumeDim>&&) = default;
  Block<VolumeDim>& operator=(const Block<VolumeDim>&) = delete;
  Block<VolumeDim>& operator=(Block<VolumeDim>&&) = default;

  const EmbeddingMap<VolumeDim, VolumeDim>& embedding_map() const noexcept {
    return *embedding_map_;
  }

  /// A unique identifier for the Block that is in the range
  /// [0, number_of_blocks -1] where number_of_blocks is the number
  /// of Blocks that cover the computational domain.
  size_t id() const noexcept { return id_; }

  /// Information about the neighboring Blocks.
  const std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>&
  neighbors() const noexcept {
    return neighbors_;
  }

  /// The directions of the faces of the Block that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const
      noexcept {
    return external_boundaries_;
  }

  /// Serialization for Charm++
  void pup(PUP::er& p);  // NOLINT

 private:
  std::unique_ptr<EmbeddingMap<VolumeDim, VolumeDim>> embedding_map_;
  size_t id_;
  std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>> neighbors_;
  std::unordered_set<Direction<VolumeDim>> external_boundaries_;
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Block<VolumeDim>& block);

template <size_t VolumeDim>
bool operator==(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept;
