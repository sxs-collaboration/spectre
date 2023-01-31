// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BlockLogicalCoordinates.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Domain.hpp"  // IWYU pragma: keep
#include "Domain/Structure/BlockId.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
// Define this alias so we don't need to keep typing this monster.
template <size_t Dim>
using block_logical_coord_holder =
    std::optional<IdPair<domain::BlockId,
                         tnsr::I<double, Dim, typename ::Frame::BlockLogical>>>;
using functions_of_time_type = std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
}  // namespace

template <size_t Dim, typename Frame>
std::vector<block_logical_coord_holder<Dim>> block_logical_coordinates(
    const Domain<Dim>& domain, const tnsr::I<DataVector, Dim, Frame>& x,
    const double time, const functions_of_time_type& functions_of_time) {
  const size_t num_pts = get<0>(x).size();
  std::vector<block_logical_coord_holder<Dim>> block_coord_holders(num_pts);
  for (size_t s = 0; s < num_pts; ++s) {
    tnsr::I<double, Dim, Frame> x_frame(0.0);
    for (size_t d = 0; d < Dim; ++d) {
      x_frame.get(d) = x.get(d)[s];
    }
    tnsr::I<double, Dim, typename ::Frame::BlockLogical> x_logical{};
    // Check which block this point is in. Each point will be in one
    // and only one block, unless it is on a shared boundary.  In that
    // case, choose the first matching block (and this block will have
    // the smallest block_id).
    for (const auto& block : domain.blocks()) {
      if (block.is_time_dependent()) {
        if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
          // Point is in the inertial frame, so we need to map to the grid
          // frame and then the logical frame.
          const auto moving_inv =
              block.moving_mesh_grid_to_inertial_map().inverse(
                  x_frame, time, functions_of_time);
          if (not moving_inv.has_value()) {
            continue;
          }
          // logical to grid map is time-independent.
          const auto inv = block.moving_mesh_logical_to_grid_map().inverse(
              moving_inv.value());
          if (inv.has_value()) {
            x_logical = inv.value();
          } else {
            continue;  // Not in this block
          }
        } else if constexpr (std::is_same_v<Frame, ::Frame::Distorted>) {
          // Point is in the distorted frame, so we need to map to the grid
          // frame and then the logical frame.
          if (not block.has_distorted_frame()) {
            // Note that block.has_distorted_frame() can be different for
            // different Blocks.  However, the template parameter Frame is
            // compile-time and is the same for all Blocks.
            //
            // Explanation of the logic here:
            // 1. Recall that block_logical_coordinates loops through all the
            //    Blocks, and skips all the Blocks except for the first Block
            //    it finds that contains the point x.
            // 2. If Frame is ::Frame::Distorted but
            //    block.has_distorted_frame() is false, then this block
            //    cannot contain the point x. Therefore, we should simply
            //    skip this block.  If it turns out that no blocks contain
            //    the point x, then we will get an error later.
            //    (Note that our primary use case for ::Frame::Distorted is to
            //    find an apparent horizon in the distorted frame. In that
            //    case, only the Blocks near a horizon have a distorted frame
            //    because only those Blocks have distortion maps. Thus,
            //    the Blocks that are skipped here are those that are far
            //    from horizons).
            continue; // Not in this block
          }
          const auto moving_inv =
              block.moving_mesh_grid_to_distorted_map().inverse(
                  x_frame, time, functions_of_time);
          if (not moving_inv.has_value()) {
            continue; // Not in this block
          }
          // logical to grid map is time-independent.
          const auto inv = block.moving_mesh_logical_to_grid_map().inverse(
              moving_inv.value());
          if (inv.has_value()) {
            x_logical = inv.value();
          } else {
            continue;  // Not in this block
          }
        } else {
          // frame is different than ::Frame::Inertial or ::Frame::Distorted.
          // Currently 'time' is unused in this branch.
          // To make the compiler happy, need to trick it to think that
          // 'time' is used.
          (void) time;
          // Currently we only support Grid, Distorted and Inertial
          // frames in the block, so make sure Frame is
          // ::Frame::Grid. (The Inertial and Distorted cases were
          // handled above.)
          static_assert(std::is_same_v<Frame, ::Frame::Grid>,
                        "Cannot convert from given frame to Grid frame");

          // Point is in the grid frame, just map to logical frame.
          const auto inv =
              block.moving_mesh_logical_to_grid_map().inverse(x_frame);
          if (inv.has_value()) {
            x_logical = inv.value();
          } else {
            continue;  // Not in this block
          }
        }
      } else {  // not block.is_time_dependent()
        if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
          const auto inv = block.stationary_map().inverse(x_frame);
          if (inv.has_value()) {
            x_logical = inv.value();
          } else {
            continue;  // Not in this block
          }
        } else {
          // If the map is time-independent, then the grid, distorted, and
          // inertial frames are the same.  So if we are in the grid
          // or distorted frames, convert to the inertial frame
          // (this conversion is just a type conversion).
          // Otherwise throw a static_assert.
          static_assert(std::is_same_v<Frame, ::Frame::Grid> or
                            std::is_same_v<Frame, ::Frame::Distorted>,
                        "Cannot convert from given frame to Inertial frame");
          tnsr::I<double, Dim, ::Frame::Inertial> x_inertial(0.0);
          for (size_t d = 0; d < Dim; ++d) {
            x_inertial.get(d) = x_frame.get(d);
          }
          const auto inv = block.stationary_map().inverse(x_inertial);
          if (inv.has_value()) {
            x_logical = inv.value();
          } else {
            continue;  // Not in this block
          }
        }
      }
      bool is_contained = true;
      for (size_t d = 0; d < Dim; ++d) {
        // Map inverses may report logical coordinates outside [-1, 1] due to
        // numerical roundoff error. In that case we clamp them to -1 or 1 so
        // that a consistent block is chosen here independent of roundoff error.
        // Without this correction, points on block boundaries where both blocks
        // report logical coordinates outside [-1, 1] by roundoff error would
        // not be assigned to any block at all, even though they lie in the
        // domain.
        if (equal_within_roundoff(x_logical.get(d), 1.0)) {
          x_logical.get(d) = 1.0;
          continue;
        }
        if (equal_within_roundoff(x_logical.get(d), -1.0)) {
          x_logical.get(d) = -1.0;
          continue;
        }
        is_contained = is_contained and abs(x_logical.get(d)) <= 1.0;
      }
      if (is_contained) {
        // Point is in this block.  Don't bother checking subsequent
        // blocks.
        block_coord_holders[s] =
            make_id_pair(domain::BlockId(block.id()), std::move(x_logical));
        break;
      }
    }
  }
  return block_coord_holders;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template std::vector<block_logical_coord_holder<DIM(data)>>                  \
  block_logical_coordinates(                                                   \
      const Domain<DIM(data)>& domain,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& x, const double time, \
      const functions_of_time_type& functions_of_time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef FRAME
#undef DIM
#undef INSTANTIATE
