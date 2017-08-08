// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Block.hpp"

#include <ostream>

#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
Block<VolumeDim>::Block(
    const EmbeddingMap<VolumeDim, VolumeDim>& embedding_map, const size_t id,
    std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>&&
        neighbors)
    : embedding_map_(embedding_map.get_clone()),
      id_(id),
      neighbors_(std::move(neighbors)) {
  // Loop over Directions to search which Directions were not set to neighbors_,
  // set these Directions to external_boundaries_.
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    if (neighbors_.find(direction) == neighbors_.end()) {
      external_boundaries_.emplace(std::move(direction));
    }
  }
}

template <size_t VolumeDim>
void Block<VolumeDim>::pup(PUP::er& p) {
  p | embedding_map_;
  p | id_;
  p | neighbors_;
  p | external_boundaries_;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Block<VolumeDim>& block) {
  os << "Block " << block.id() << ":\n";
  os << "Neighbors:\n";
  for (const auto& direction_to_neighbors : block.neighbors()) {
    os << direction_to_neighbors.first << ": " << direction_to_neighbors.second
       << "\n";
  }
  os << "External boundaries: " << block.external_boundaries() << "\n";
  return os;
}

template <size_t VolumeDim>
bool operator==(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept {
  return (lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
          lhs.external_boundaries() == rhs.external_boundaries());
}

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
template class Block<1>;
template class Block<2>;
template class Block<3>;

template std::ostream& operator<<(std::ostream&, const Block<1>&);
template std::ostream& operator<<(std::ostream&, const Block<2>&);
template std::ostream& operator<<(std::ostream&, const Block<3>&);

template bool operator==(const Block<1>&, const Block<1>&);
template bool operator==(const Block<2>&, const Block<2>&);
template bool operator==(const Block<3>&, const Block<3>&);

template bool operator!=(const Block<1>&, const Block<1>&);
template bool operator!=(const Block<2>&, const Block<2>&);
template bool operator!=(const Block<3>&, const Block<3>&);
