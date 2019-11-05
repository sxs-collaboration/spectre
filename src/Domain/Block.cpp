// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Block.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

template <size_t VolumeDim>
Block<VolumeDim>::Block(
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, Frame::Inertial,
                                              VolumeDim>>&& map,
    const size_t id,
    DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors) noexcept
    : map_(std::move(map)), id_(id), neighbors_(std::move(neighbors)) {
  // Loop over Directions to search which Directions were not set to neighbors_,
  // set these Directions to external_boundaries_.
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    if (neighbors_.find(direction) == neighbors_.end()) {
      external_boundaries_.emplace(std::move(direction));
    }
  }
}

template <size_t VolumeDim>
void Block<VolumeDim>::pup(PUP::er& p) noexcept {
  p | map_;
  p | id_;
  p | neighbors_;
  p | external_boundaries_;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Block<VolumeDim>& block) noexcept {
  os << "Block " << block.id() << ":\n";
  os << "Neighbors: " << block.neighbors() << '\n';
  os << "External boundaries: " << block.external_boundaries() << '\n';
  return os;
}

template <size_t VolumeDim>
bool operator==(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept {
  return lhs.id() == rhs.id() and lhs.neighbors() == rhs.neighbors() and
         lhs.external_boundaries() == rhs.external_boundaries() and
         lhs.coordinate_map() == rhs.coordinate_map();
}

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                          \
  template class Block<GET_DIM(data)>;                                  \
  template std::ostream& operator<<(std::ostream& os,                   \
                                    const Block<GET_DIM(data)>& block); \
  template bool operator==(const Block<GET_DIM(data)>& lhs,             \
                           const Block<GET_DIM(data)>& rhs) noexcept;   \
  template bool operator!=(const Block<GET_DIM(data)>& lhs,             \
                           const Block<GET_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
