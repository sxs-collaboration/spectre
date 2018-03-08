// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/VariablesHelpers.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"

namespace OrientVariablesOnSlice_detail {

std::vector<size_t> oriented_offset(
    const Index<1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor) noexcept {
  // The slice of a 2D mesh is either aligned or anti-aligned
  const Direction<2> my_slice_axis =
      (0 == sliced_dim ? Direction<2>::upper_eta() : Direction<2>::upper_xi());
  const Direction<2> neighbor_slice_axis =
      orientation_of_neighbor(my_slice_axis);
  std::vector<size_t> oriented_offsets(slice_extents.product());
  std::iota(oriented_offsets.begin(), oriented_offsets.end(), 0);

  if (neighbor_slice_axis.side() == Side::Lower) {
    std::reverse(oriented_offsets.begin(), oriented_offsets.end());
  }
  return oriented_offsets;
}

// The slice of a 3D mesh can have eight different data-storage orders
// depending upon the relative orientation of the neighbor.  These are
// determined by whether the new data-storage order varies fastest by the
// lowest dim or the highest dim, and by whether each axis is aligned or
// anti-aligned.
std::vector<size_t> oriented_offset(
    const Index<2>& slice_extents, const size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor) noexcept {
  const std::array<size_t, 2> dims_of_slice =
      (0 == sliced_dim ? make_array(1_st, 2_st)
                       : (1 == sliced_dim) ? make_array(0_st, 2_st)
                                           : make_array(0_st, 1_st));
  const bool transpose_needed = (orientation_of_neighbor(dims_of_slice[0]) >
                                 orientation_of_neighbor(dims_of_slice[1]));
  const Direction<3> neighbor_first_axis =
      orientation_of_neighbor(Direction<3>(dims_of_slice[0], Side::Upper));
  const Direction<3> neighbor_second_axis =
      orientation_of_neighbor(Direction<3>(dims_of_slice[1], Side::Upper));
  const bool neighbor_first_axis_flipped =
      (Side::Lower == neighbor_first_axis.side());
  const bool neighbor_second_axis_flipped =
      (Side::Lower == neighbor_second_axis.side());

  std::vector<size_t> oriented_offsets(slice_extents.product());
  const size_t num_pts_1 = slice_extents[0];
  const size_t num_pts_2 = slice_extents[1];
  if (transpose_needed) {
    if (neighbor_first_axis_flipped) {
      if (neighbor_second_axis_flipped) {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] =
                (num_pts_2 - 1 - i2) + num_pts_2 * (num_pts_1 - 1 - i1);
          }
        }
      } else {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] =
                i2 + num_pts_2 * (num_pts_1 - 1 - i1);
          }
        }
      }
    } else {
      if (neighbor_second_axis_flipped) {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] =
                (num_pts_2 - 1 - i2) + num_pts_2 * i1;
          }
        }
      } else {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] = i2 + num_pts_2 * i1;
          }
        }
      }
    }
  } else {
    if (neighbor_first_axis_flipped) {
      if (neighbor_second_axis_flipped) {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] =
                (num_pts_1 - 1 - i1) + num_pts_1 * (num_pts_2 - 1 - i2);
          }
        }
      } else {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] =
                (num_pts_1 - 1 - i1) + num_pts_1 * i2;
          }
        }
      }
    } else {
      if (neighbor_second_axis_flipped) {
        for (size_t i2 = 0; i2 < num_pts_2; ++i2) {
          for (size_t i1 = 0; i1 < num_pts_1; ++i1) {
            oriented_offsets[i1 + num_pts_1 * i2] =
                i1 + num_pts_1 * (num_pts_2 - 1 - i2);
          }
        }
      } else {
        std::iota(oriented_offsets.begin(), oriented_offsets.end(), 0);
      }
    }
  }
  return oriented_offsets;
}
}  // namespace OrientVariablesOnSlice_detail
