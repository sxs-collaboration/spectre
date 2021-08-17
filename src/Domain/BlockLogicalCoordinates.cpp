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
    const double time,
    const functions_of_time_type& functions_of_time) noexcept {
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
        } else {  // frame is different than ::Frame::Inertial
          // Currently 'time' is unused in this branch.
          // To make the compiler happy, need to trick it to think that
          // 'time' is used.
          (void) time;
          // Currently we only support Grid and Inertial frames in the
          // block, so make sure Frame is ::Frame::Grid. (The
          // Inertial case was handled above.)
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
          // If the map is time-independent, then the grid and
          // inertial frames are the same.  So if we are in the grid frame,
          // convert to the inertial frame.  Otherwise throw a static_assert.
          // Once we support more frames (e.g. distorted) this logic will
          // change.
          static_assert(std::is_same_v<Frame, ::Frame::Grid>,
                        "Cannot convert from given frame to Grid frame");
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
        // Assumes that logical coordinates go from -1 to +1 in each
        // dimension.
        is_contained = is_contained and x_logical.get(d) >= -1.0 and
                       x_logical.get(d) <= 1.0;
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
      const functions_of_time_type& functions_of_time) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (::Frame::Inertial, ::Frame::Grid))

#undef FRAME
#undef DIM
#undef INSTANTIATE
