// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Domain.hpp"

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/DomainHelpers.hpp"

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame>::Domain(
    std::vector<Block<VolumeDim, TargetFrame>> blocks)
    : blocks_(std::move(blocks)) {}

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame>::Domain(
    std::vector<std::unique_ptr<
        CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>>
        maps,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    const std::vector<std::vector<size_t>>& identifications) {
  std::vector<
      std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(corners_of_all_blocks,
                                     &neighbors_of_all_blocks);
  set_periodic_boundaries<VolumeDim>(identifications, corners_of_all_blocks,
                                     &neighbors_of_all_blocks);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    blocks_.emplace_back(std::move(maps[i]), i,
                         std::move(neighbors_of_all_blocks[i]));
  }
}

template <size_t VolumeDim, typename TargetFrame>
std::ostream& operator<<(std::ostream& os,
                         const Domain<VolumeDim, TargetFrame>& d) {
  const auto& blocks = d.blocks();
  os << "Domain with " << blocks.size() << " blocks:\n";
  for (const auto& block : blocks) {
    os << block << "\n";
  }
  return os;
}

template <size_t VolumeDim, typename TargetFrame>
void Domain<VolumeDim, TargetFrame>::pup(PUP::er& p) {
  p | blocks_;
}

// Explicit Instantiations
template class Domain<1, Frame::Grid>;
template class Domain<2, Frame::Grid>;
template class Domain<3, Frame::Grid>;
template class Domain<1, Frame::Inertial>;
template class Domain<2, Frame::Inertial>;
template class Domain<3, Frame::Inertial>;

template std::ostream& operator<<(std::ostream& os,
                                  const Domain<1, Frame::Grid>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Domain<2, Frame::Grid>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Domain<3, Frame::Grid>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Domain<1, Frame::Inertial>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Domain<2, Frame::Inertial>& block);
template std::ostream& operator<<(std::ostream& os,
                                  const Domain<3, Frame::Inertial>& block);
