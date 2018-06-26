// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "BlockLogicalCoordinates.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/Domain.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
// Define this alias so we don't need to keep typing this monster.
template <size_t Dim>
using block_logical_coord_holder =
    IdPair<domain::BlockId, tnsr::I<double, Dim, typename ::Frame::Logical>>;
}  // namespace

template <size_t Dim, typename Frame>
std::vector<block_logical_coord_holder<Dim>> block_logical_coordinates(
    const Domain<Dim, Frame>& domain,
    const tnsr::I<DataVector, Dim, Frame>& x) noexcept {
  const size_t num_pts = get<0>(x).size();
  std::vector<block_logical_coord_holder<Dim>> block_coord_holders(num_pts);
  std::vector<tnsr::I<double, Dim, Frame>> points_with_no_block;
  for (size_t s = 0; s < num_pts; ++s) {
    tnsr::I<double, Dim, Frame> x_frame(0.0);
    for (size_t d = 0; d < Dim; ++d) {
      x_frame.get(d) = x.get(d)[s];
    }
    auto& x_logical = block_coord_holders[s].data;
    // Check which block this point is in. Each point will be in one
    // and only one block, unless it is on a shared boundary.  In that
    // case, choose the first matching block (and this block will have
    // the smallest block_id).
    bool found_block = false;
    for (const auto& block : domain.blocks()) {
      const auto inv = block.coordinate_map().inverse(x_frame);
      if(inv) {
        x_logical = inv.get();
      } else {
        continue; // Not in this block
      }
      bool is_contained = true;
      for (size_t d = 0; d < Dim; ++d) {
        // Assumes that logical coordinates go from -1 to +1 in each
        // dimension.
        is_contained = is_contained and x_logical.get(d) >= -1.0 and
                       x_logical.get(d) <= 1.0;
      }
      if (is_contained) {
        // Point is in this block.  Don't bother checking subsequent
        // blocks.
        block_coord_holders[s].id = domain::BlockId(block.id());
        found_block = true;
        break;
      }
    }
    if (not found_block) {
      points_with_no_block.emplace_back(std::move(x_frame));
    }
  }
  if (not points_with_no_block.empty()) {
    ERROR("Found points that are not in any block\n: x_frame = "
          << points_with_no_block);
  }
  return block_coord_holders;
}

// Explicit instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                  \
  template std::vector<block_logical_coord_holder<DIM(data)>> \
  block_logical_coordinates(                                  \
      const Domain<DIM(data), FRAME(data)>& domain,           \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& x) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (Frame::Distorted, Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
/// \endcond
