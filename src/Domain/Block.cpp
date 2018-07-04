// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Block.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame

namespace domain {
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
  ::operator<<(os << "External boundaries: ", block.external_boundaries())
      << "\n";
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

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template class Block<DIM(data), FRAME(data)>;                                \
  template std::ostream& operator<<(                                           \
      std::ostream& os, const Block<DIM(data), FRAME(data)>& block);           \
  template bool operator==(const Block<DIM(data), FRAME(data)>& lhs,           \
                           const Block<DIM(data), FRAME(data)>& rhs) noexcept; \
  template bool operator!=(const Block<DIM(data), FRAME(data)>& lhs,           \
                           const Block<DIM(data), FRAME(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
}  // namespace domain
