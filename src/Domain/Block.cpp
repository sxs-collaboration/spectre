// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Block.hpp"

template <size_t VolumeDim, typename TargetFrame>
Block<VolumeDim, TargetFrame>::Block(
    std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>&&
        map,
    const size_t id,
    std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>
        neighbors)
    : map_(std::move(map)), id_(id), neighbors_(std::move(neighbors)) {
  // Loop over Directions to search which Directions were not set to neighbors_,
  // set these Directions to external_boundaries_.
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    if (neighbors_.find(direction) == neighbors_.end()) {
      external_boundaries_.emplace(std::move(direction));
    }
  }
}

template <size_t VolumeDim, typename TargetFrame>
void Block<VolumeDim, TargetFrame>::pup(PUP::er& p) {
  p | map_;
  p | id_;
  p | neighbors_;
  p | external_boundaries_;
}

template <size_t VolumeDim, typename TargetFrame>
std::ostream& operator<<(std::ostream& os,
                         const Block<VolumeDim, TargetFrame>& block) {
  os << "Block " << block.id() << ":\n";
  os << "Neighbors:\n";
  for (const auto& direction_to_neighbors : block.neighbors()) {
    os << direction_to_neighbors.first << ": " << direction_to_neighbors.second
       << "\n";
  }
  os << "External boundaries: " << block.external_boundaries() << "\n";
  return os;
}

template <size_t VolumeDim, typename TargetFrame>
bool operator==(const Block<VolumeDim, TargetFrame>& lhs,
                const Block<VolumeDim, TargetFrame>& rhs) noexcept {
  return lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
         lhs.external_boundaries() == rhs.external_boundaries() and
         lhs.coordinate_map() == rhs.coordinate_map();
}

template <size_t VolumeDim, typename TargetFrame>
bool operator!=(const Block<VolumeDim, TargetFrame>& lhs,
                const Block<VolumeDim, TargetFrame>& rhs) noexcept {
  return not(lhs == rhs);
}

template class Block<1, Frame::Grid>;
template class Block<2, Frame::Grid>;
template class Block<3, Frame::Grid>;
template class Block<1, Frame::Inertial>;
template class Block<2, Frame::Inertial>;
template class Block<3, Frame::Inertial>;

template std::ostream& operator<<(std::ostream& os,
                                  const Block<1, Frame::Grid>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Block<2, Frame::Grid>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Block<3, Frame::Grid>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Block<1, Frame::Inertial>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Block<2, Frame::Inertial>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Block<3, Frame::Inertial>& block);

template bool operator==(const Block<1, Frame::Grid>& lhs,
                         const Block<1, Frame::Grid>& rhs) noexcept;
template bool operator==(const Block<2, Frame::Grid>& lhs,
                         const Block<2, Frame::Grid>& rhs) noexcept;
template bool operator==(const Block<3, Frame::Grid>& lhs,
                         const Block<3, Frame::Grid>& rhs) noexcept;
template bool operator==(const Block<1, Frame::Inertial>& lhs,
                         const Block<1, Frame::Inertial>& rhs) noexcept;
template bool operator==(const Block<2, Frame::Inertial>& lhs,
                         const Block<2, Frame::Inertial>& rhs) noexcept;
template bool operator==(const Block<3, Frame::Inertial>& lhs,
                         const Block<3, Frame::Inertial>& rhs) noexcept;

template bool operator!=(const Block<1, Frame::Grid>& lhs,
                         const Block<1, Frame::Grid>& rhs) noexcept;
template bool operator!=(const Block<2, Frame::Grid>& lhs,
                         const Block<2, Frame::Grid>& rhs) noexcept;
template bool operator!=(const Block<3, Frame::Grid>& lhs,
                         const Block<3, Frame::Grid>& rhs) noexcept;
template bool operator!=(const Block<1, Frame::Inertial>& lhs,
                         const Block<1, Frame::Inertial>& rhs) noexcept;
template bool operator!=(const Block<2, Frame::Inertial>& lhs,
                         const Block<2, Frame::Inertial>& rhs) noexcept;
template bool operator!=(const Block<3, Frame::Inertial>& lhs,
                         const Block<3, Frame::Inertial>& rhs) noexcept;
