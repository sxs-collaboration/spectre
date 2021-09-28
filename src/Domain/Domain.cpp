// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Domain.hpp"

#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"      // IWYU pragma: keep
#include "Domain/Structure/DirectionMap.hpp"   // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

template <size_t VolumeDim>
Domain<VolumeDim>::Domain(std::vector<Block<VolumeDim>> blocks)
    : blocks_(std::move(blocks)) {}

template <size_t VolumeDim>
Domain<VolumeDim>::Domain(
    std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>>
        maps,
    std::vector<DirectionMap<
        VolumeDim,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        boundary_conditions) {
  ASSERT(
      boundary_conditions.size() == maps.size() or boundary_conditions.empty(),
      "There must be either one set of boundary conditions per block or none "
      "at all specified.");

  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(&neighbors_of_all_blocks, maps);
  for (size_t i = 0; i < maps.size(); i++) {
    if (boundary_conditions.empty()) {
      blocks_.emplace_back(std::move(maps[i]), i,
                           std::move(neighbors_of_all_blocks[i]));
    } else {
      blocks_.emplace_back(std::move(maps[i]), i,
                           std::move(neighbors_of_all_blocks[i]),
                           std::move(boundary_conditions[i]));
    }
  }
}

template <size_t VolumeDim>
Domain<VolumeDim>::Domain(
    std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>>
        maps,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    const std::vector<PairOfFaces>& identifications,
    std::vector<DirectionMap<
        VolumeDim,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        boundary_conditions) {
  ASSERT(
      maps.size() == corners_of_all_blocks.size(),
      "Must pass same number of maps as block corner sets, but maps.size() == "
          << maps.size() << " and corners_of_all_blocks.size() == "
          << corners_of_all_blocks.size());
  ASSERT(
      boundary_conditions.size() == maps.size() or boundary_conditions.empty(),
      "There must be either one set of boundary conditions per block or none "
      "at all specified.");
  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(&neighbors_of_all_blocks,
                                     corners_of_all_blocks);
  set_identified_boundaries<VolumeDim>(identifications, corners_of_all_blocks,
                                       &neighbors_of_all_blocks);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    if (boundary_conditions.empty()) {
      blocks_.emplace_back(std::move(maps[i]), i,
                           std::move(neighbors_of_all_blocks[i]));
    } else {
      blocks_.emplace_back(std::move(maps[i]), i,
                           std::move(neighbors_of_all_blocks[i]),
                           std::move(boundary_conditions[i]));
    }
  }
}

template <size_t VolumeDim>
void Domain<VolumeDim>::inject_time_dependent_map_for_block(
    const size_t block_id,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
        moving_mesh_inertial_map) {
  ASSERT(block_id < blocks_.size(),
         "The block id " << block_id
                         << " larger than the total number of blocks, "
                         << blocks_.size());
  blocks_[block_id].inject_time_dependent_map(
      std::move(moving_mesh_inertial_map));
}

template <size_t VolumeDim>
bool operator==(const Domain<VolumeDim>& lhs, const Domain<VolumeDim>& rhs) {
  return lhs.blocks() == rhs.blocks();
}

template <size_t VolumeDim>
bool operator!=(const Domain<VolumeDim>& lhs, const Domain<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Domain<VolumeDim>& d) {
  const auto& blocks = d.blocks();
  os << "Domain with " << blocks.size() << " blocks:\n";
  for (const auto& block : blocks) {
    os << block << '\n';
  }
  return os;
}

template <size_t VolumeDim>
void Domain<VolumeDim>::pup(PUP::er& p) {
  p | blocks_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                              \
  template class Domain<DIM(data)>;                       \
  template bool operator==(const Domain<DIM(data)>& lhs,  \
                           const Domain<DIM(data)>& rhs); \
  template bool operator!=(const Domain<DIM(data)>& lhs,  \
                           const Domain<DIM(data)>& rhs); \
  template std::ostream& operator<<(std::ostream& os,     \
                                    const Domain<DIM(data)>& d);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef FRAME
#undef INSTANTIATE

// Some compilers (clang 3.9.1) don't instantiate the default argument
// to the second Domain constructor.
template class std::vector<PairOfFaces>;
