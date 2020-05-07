// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Domain.hpp"

#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "Domain/BlockNeighbor.hpp"                 // IWYU pragma: keep
#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"                     // IWYU pragma: keep
#include "Domain/DirectionMap.hpp"                  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

template <size_t VolumeDim>
Domain<VolumeDim>::Domain(std::vector<Block<VolumeDim>> blocks) noexcept
    : blocks_(std::move(blocks)) {}

template <size_t VolumeDim>
Domain<VolumeDim>::Domain(
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>>>
        maps,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    const std::vector<PairOfFaces>& identifications) noexcept {
  ASSERT(
      maps.size() == corners_of_all_blocks.size(),
      "Must pass same number of maps as block corner sets, but maps.size() == "
          << maps.size() << " and corners_of_all_blocks.size() == "
          << corners_of_all_blocks.size());
  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(corners_of_all_blocks,
                                     &neighbors_of_all_blocks);
  set_identified_boundaries<VolumeDim>(identifications, corners_of_all_blocks,
                                       &neighbors_of_all_blocks);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    blocks_.emplace_back(std::move(maps[i]), i,
                         std::move(neighbors_of_all_blocks[i]));
  }
}

template <size_t VolumeDim>
void Domain<VolumeDim>::inject_time_dependent_map_for_block(
    const size_t block_id,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
        moving_mesh_inertial_map) noexcept {
  ASSERT(block_id < blocks_.size(),
         "The block id " << block_id
                         << " larger than the total number of blocks, "
                         << blocks_.size());
  blocks_[block_id].inject_time_dependent_map(
      std::move(moving_mesh_inertial_map));
}

template <size_t VolumeDim>
bool operator==(const Domain<VolumeDim>& lhs,
                const Domain<VolumeDim>& rhs) noexcept {
  return lhs.blocks() == rhs.blocks();
}

template <size_t VolumeDim>
bool operator!=(const Domain<VolumeDim>& lhs,
                const Domain<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Domain<VolumeDim>& d) noexcept {
  const auto& blocks = d.blocks();
  os << "Domain with " << blocks.size() << " blocks:\n";
  for (const auto& block : blocks) {
    os << block << '\n';
  }
  return os;
}

template <size_t VolumeDim>
void Domain<VolumeDim>::pup(PUP::er& p) noexcept {
  p | blocks_;
}

/// \cond HIDDEN_SYMBOLS
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                       \
  template class Domain<DIM(data)>;                                \
  template bool operator==(const Domain<DIM(data)>& lhs,           \
                           const Domain<DIM(data)>& rhs) noexcept; \
  template bool operator!=(const Domain<DIM(data)>& lhs,           \
                           const Domain<DIM(data)>& rhs) noexcept; \
  template std::ostream& operator<<(std::ostream& os,              \
                                    const Domain<DIM(data)>& d);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef FRAME
#undef INSTANTIATE
/// \endcond

// Some compilers (clang 3.9.1) don't instantiate the default argument
// to the second Domain constructor.
template class std::vector<PairOfFaces>;
