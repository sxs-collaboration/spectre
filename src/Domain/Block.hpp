// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Block.

#pragma once

#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>

#include "Domain/BlockNeighbor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/EmbeddingMaps/CoordinateMap.hpp"
#include "Utilities/StdHelpers.hpp"

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
/// \tparam Map the type of CoordinateMap the Block will hold
template <typename Map>
class Block {
 public:
  static constexpr auto dim = Map::dim;
  /// \param embedding_map the EmbeddingMap.
  /// \param id a unique ID.
  /// \param neighbors info about the Blocks that share a codimension 1
  /// boundary with this Block.
  Block(Map embedding_map, size_t id,
        std::unordered_map<Direction<dim>, BlockNeighbor<dim>>&& neighbors);

  Block() = default;
  ~Block() = default;
  Block(const Block&) = delete;
  Block(Block&&) = default;
  Block& operator=(const Block&) = delete;
  Block& operator=(Block&&) = default;

  const Map& embedding_map() const noexcept { return embedding_map_; }

  /// A unique identifier for the Block that is in the range
  /// [0, number_of_blocks -1] where number_of_blocks is the number
  /// of Blocks that cover the computational domain.
  size_t id() const noexcept { return id_; }

  /// Information about the neighboring Blocks.
  const std::unordered_map<Direction<dim>, BlockNeighbor<dim>>& neighbors()
      const noexcept {
    return neighbors_;
  }

  /// The directions of the faces of the Block that are external boundaries.
  const std::unordered_set<Direction<dim>>& external_boundaries() const
      noexcept {
    return external_boundaries_;
  }

  /// Serialization for Charm++
  void pup(PUP::er& p);  // NOLINT

 private:
  Map embedding_map_;
  size_t id_{0};
  std::unordered_map<Direction<dim>, BlockNeighbor<dim>> neighbors_;
  std::unordered_set<Direction<dim>> external_boundaries_;
};

template <typename Map>
Block<Map>::Block(
    Map embedding_map, const size_t id,
    std::unordered_map<Direction<dim>, BlockNeighbor<dim>>&& neighbors)
    : embedding_map_(std::move(embedding_map)),
      id_(id),
      neighbors_(std::move(neighbors)) {
  // Loop over Directions to search which Directions were not set to neighbors_,
  // set these Directions to external_boundaries_.
  for (const auto& direction : Direction<dim>::all_directions()) {
    if (neighbors_.find(direction) == neighbors_.end()) {
      external_boundaries_.emplace(std::move(direction));
    }
  }
}

template <typename Map>
void Block<Map>::pup(PUP::er& p) {
  p | embedding_map_;
  p | id_;
  p | neighbors_;
  p | external_boundaries_;
}

template <typename Map>
std::ostream& operator<<(std::ostream& os, const Block<Map>& block) {
  os << "Block " << block.id() << ":\n";
  os << "Neighbors:\n";
  for (const auto& direction_to_neighbors : block.neighbors()) {
    os << direction_to_neighbors.first << ": " << direction_to_neighbors.second
       << "\n";
  }
  os << "External boundaries: " << block.external_boundaries() << "\n";
  return os;
}

template <typename Map>
bool operator==(const Block<Map>& lhs, const Block<Map>& rhs) noexcept {
  return (lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
          lhs.external_boundaries() == rhs.external_boundaries());
}

template <typename Map>
bool operator!=(const Block<Map>& lhs, const Block<Map>& rhs) noexcept {
  return not(lhs == rhs);
}
