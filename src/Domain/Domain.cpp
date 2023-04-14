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
#include "Utilities/Algorithm.hpp"
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
    std::unordered_map<std::string, ExcisionSphere<VolumeDim>> excision_spheres,
    std::vector<std::string> block_names,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups)
    : excision_spheres_(std::move(excision_spheres)),
      block_groups_(std::move(block_groups)) {
  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(&neighbors_of_all_blocks, maps);
  ASSERT(block_names.empty() or block_names.size() == maps.size(),
         "Expected " << maps.size() << " block names but got "
                     << block_names.size() << ".");
  for (size_t i = 0; i < maps.size(); i++) {
    blocks_.emplace_back(std::move(maps[i]), i,
                         std::move(neighbors_of_all_blocks[i]),
                         block_names.empty() ? "" : block_names[i]);
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
    std::unordered_map<std::string, ExcisionSphere<VolumeDim>> excision_spheres,
    std::vector<std::string> block_names,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups)
    : excision_spheres_(std::move(excision_spheres)),
      block_groups_(std::move(block_groups)) {
  ASSERT(
      maps.size() == corners_of_all_blocks.size(),
      "Must pass same number of maps as block corner sets, but maps.size() == "
          << maps.size() << " and corners_of_all_blocks.size() == "
          << corners_of_all_blocks.size());
  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(&neighbors_of_all_blocks,
                                     corners_of_all_blocks);
  set_identified_boundaries<VolumeDim>(identifications, corners_of_all_blocks,
                                       &neighbors_of_all_blocks);
  ASSERT(block_names.empty() or block_names.size() == maps.size(),
         "Expected " << maps.size() << " block names but got "
                     << block_names.size() << ".");
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    blocks_.emplace_back(std::move(maps[i]), i,
                         std::move(neighbors_of_all_blocks[i]),
                         block_names.empty() ? "" : block_names[i]);
  }
}

template <size_t VolumeDim>
void Domain<VolumeDim>::inject_time_dependent_map_for_block(
    const size_t block_id,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
        moving_mesh_grid_to_inertial_map,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>>
        moving_mesh_grid_to_distorted_map,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, VolumeDim>>
        moving_mesh_distorted_to_inertial_map) {
  ASSERT(block_id < blocks_.size(),
         "The block id " << block_id
                         << " larger than the total number of blocks, "
                         << blocks_.size());
  blocks_[block_id].inject_time_dependent_map(
      std::move(moving_mesh_grid_to_inertial_map),
      std::move(moving_mesh_grid_to_distorted_map),
      std::move(moving_mesh_distorted_to_inertial_map));
}

template <size_t VolumeDim>
bool Domain<VolumeDim>::is_time_dependent() const {
  return alg::any_of(blocks_, [](const Block<VolumeDim>& block) {
    return block.is_time_dependent();
  });
}

template <size_t VolumeDim>
bool operator==(const Domain<VolumeDim>& lhs, const Domain<VolumeDim>& rhs) {
  return lhs.blocks() == rhs.blocks() and
         lhs.excision_spheres() == rhs.excision_spheres() and
         lhs.block_groups() == rhs.block_groups();
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
  os << "Excision spheres:\n" << d.excision_spheres() << '\n';
  return os;
}

template <size_t VolumeDim>
void Domain<VolumeDim>::pup(PUP::er& p) {
  size_t version = 1;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | blocks_;
    p | excision_spheres_;
  }
  if (version >= 1) {
    p | block_groups_;
  }
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
