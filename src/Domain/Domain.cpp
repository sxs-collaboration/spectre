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
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame>::Domain(
    std::vector<Block<VolumeDim, TargetFrame>> blocks) noexcept
    : blocks_(std::move(blocks)) {}

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame>::Domain(
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>>
        maps,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    const std::vector<PairOfFaces>& identifications) noexcept {
  ASSERT(maps.size() == corners_of_all_blocks.size(),
         "Must pass same number of maps as block corner sets.");
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

template <size_t VolumeDim, typename TargetFrame>
bool operator==(const Domain<VolumeDim, TargetFrame>& lhs,
                const Domain<VolumeDim, TargetFrame>& rhs) noexcept {
  return lhs.blocks() == rhs.blocks();
}

template <size_t VolumeDim, typename TargetFrame>
bool operator!=(const Domain<VolumeDim, TargetFrame>& lhs,
                const Domain<VolumeDim, TargetFrame>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t VolumeDim, typename TargetFrame>
std::ostream& operator<<(std::ostream& os,
                         const Domain<VolumeDim, TargetFrame>& d) noexcept {
  const auto& blocks = d.blocks();
  os << "Domain with " << blocks.size() << " blocks:\n";
  for (const auto& block : blocks) {
    os << block << '\n';
  }
  return os;
}

template <size_t VolumeDim, typename TargetFrame>
void Domain<VolumeDim, TargetFrame>::pup(PUP::er& p) noexcept {
  p | blocks_;
}

/// \cond HIDDEN_SYMBOLS
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                               \
  template class Domain<DIM(data), FRAME(data)>;           \
  template bool operator==(                                \
      const Domain<DIM(data), FRAME(data)>& lhs,           \
      const Domain<DIM(data), FRAME(data)>& rhs) noexcept; \
  template bool operator!=(                                \
      const Domain<DIM(data), FRAME(data)>& lhs,           \
      const Domain<DIM(data), FRAME(data)>& rhs) noexcept; \
  template std::ostream& operator<<(std::ostream& os,      \
                                    const Domain<DIM(data), FRAME(data)>& d);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
/// \endcond

// Some compilers (clang 3.9.1) don't instantiate the default argument
// to the second Domain constructor.
template class std::vector<PairOfFaces>;
