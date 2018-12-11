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

namespace {

// 1D data can be aligned or anti-aligned
std::vector<size_t> compute_offset_permutation(
    const Index<1>& extents, const bool neighbor_axis_is_aligned) noexcept {
  std::vector<size_t> oriented_offsets(extents.product());
  std::iota(oriented_offsets.begin(), oriented_offsets.end(), 0);
  if (not neighbor_axis_is_aligned) {
    std::reverse(oriented_offsets.begin(), oriented_offsets.end());
  }
  return oriented_offsets;
}

// 2D data can have 8 different data-storage orders relative to the neighbor.
// These are determined by whether the new data-storage order varies fastest by
// the lowest dim or the highest dim, and by whether each axis is aligned or
// anti-aligned.
std::vector<size_t> compute_offset_permutation(
    const Index<2>& extents, const bool neighbor_first_axis_is_aligned,
    const bool neighbor_second_axis_is_aligned,
    const bool neighbor_axes_are_transposed) noexcept {
  std::vector<size_t> oriented_offsets(extents.product());
  // Reduce the number of cases to explicitly write out by 4, by encoding the
  // (anti-)alignment of each axis as numerical factors ("offset" and "step")
  // that then contribute in identically-structured loops.
  // But doing this requires mixing positive and negative factors, so we cast
  // from size_t to int, do the work, then cast from int back to size_t.
  const auto num_pts_1 = static_cast<int>(extents[0]);
  const auto num_pts_2 = static_cast<int>(extents[1]);
  const int i1_offset = neighbor_first_axis_is_aligned ? 0 : num_pts_1 - 1;
  const int i1_step = neighbor_first_axis_is_aligned ? 1 : -1;
  const int i2_offset = neighbor_second_axis_is_aligned ? 0 : num_pts_2 - 1;
  const int i2_step = neighbor_second_axis_is_aligned ? 1 : -1;
  if (neighbor_axes_are_transposed) {
    for (int i2 = 0; i2 < num_pts_2; ++i2) {
      for (int i1 = 0; i1 < num_pts_1; ++i1) {
        // NOLINTNEXTLINE(misc-misplaced-widening-cast)
        oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2)] =
            // NOLINTNEXTLINE(misc-misplaced-widening-cast)
            static_cast<size_t>((i2_offset + i2_step * i2) +
                                num_pts_2 * (i1_offset + i1_step * i1));
      }
    }
  } else {
    for (int i2 = 0; i2 < num_pts_2; ++i2) {
      for (int i1 = 0; i1 < num_pts_1; ++i1) {
        // NOLINTNEXTLINE(misc-misplaced-widening-cast)
        oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2)] =
            // NOLINTNEXTLINE(misc-misplaced-widening-cast)
            static_cast<size_t>((i1_offset + i1_step * i1) +
                                num_pts_1 * (i2_offset + i2_step * i2));
      }
    }
  }
  return oriented_offsets;
}

// 3D data can have 48 (!) different data-storage orders relative to the
// neighbor. A factor of 6 arises from the different ways in which the three
// dimensions can be ordered from fastest to slowest varying. The remaining
// factor of 8 arises from having two possible directions (aligned or
// anti-aligned) for each of the three axes.
std::vector<size_t> compute_offset_permutation(
    const Index<3>& extents, const bool neighbor_first_axis_is_aligned,
    const bool neighbor_second_axis_is_aligned,
    const bool neighbor_third_axis_is_aligned,
    const std::array<size_t, 3>& neighbor_axis_permutation) noexcept {
  std::vector<size_t> oriented_offsets(extents.product());
  // Reduce the number of cases to explicitly write out by 8, by encoding the
  // (anti-)alignment of each axis as numerical factors ("offset" and "step")
  // that then contribute in identically-structured loops.
  // But doing this requires mixing positive and negative factors, so we cast
  // from size_t to int, do the work, then cast from int back to size_t.
  const auto num_pts_1 = static_cast<int>(extents[0]);
  const auto num_pts_2 = static_cast<int>(extents[1]);
  const auto num_pts_3 = static_cast<int>(extents[2]);
  const int i1_offset = neighbor_first_axis_is_aligned ? 0 : num_pts_1 - 1;
  const int i1_step = neighbor_first_axis_is_aligned ? 1 : -1;
  const int i2_offset = neighbor_second_axis_is_aligned ? 0 : num_pts_2 - 1;
  const int i2_step = neighbor_second_axis_is_aligned ? 1 : -1;
  const int i3_offset = neighbor_third_axis_is_aligned ? 0 : num_pts_3 - 1;
  const int i3_step = neighbor_third_axis_is_aligned ? 1 : -1;
  // The three cyclic permutations of the dimensions 0, 1, 2
  // Note that these do not necessarily lead to right-handed coordinate
  // systems, because the final "handedness" also depends on whether
  // each axis is aligned or anti-aligned.
  if (neighbor_axis_permutation == make_array(0_st, 1_st, 2_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(misc-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(misc-misplaced-widening-cast)
              static_cast<size_t>((i1_offset + i1_step * i1) +
                                  num_pts_1 * (i2_offset + i2_step * i2) +
                                  num_pts_1 * num_pts_2 *
                                      (i3_offset + i3_step * i3));
        }
      }
    }
  } else if (neighbor_axis_permutation == make_array(1_st, 2_st, 0_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(misc-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(misc-misplaced-widening-cast)
              static_cast<size_t>((i3_offset + i3_step * i3) +
                                  num_pts_3 * (i1_offset + i1_step * i1) +
                                  num_pts_3 * num_pts_1 *
                                      (i2_offset + i2_step * i2));
        }
      }
    }
  } else if (neighbor_axis_permutation == make_array(2_st, 0_st, 1_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(misc-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(misc-misplaced-widening-cast)
              static_cast<size_t>((i2_offset + i2_step * i2) +
                                  num_pts_2 * (i3_offset + i3_step * i3) +
                                  num_pts_2 * num_pts_3 *
                                      (i1_offset + i1_step * i1));
        }
      }
    }
  }
  // The three acyclic permutations
  else if (neighbor_axis_permutation == make_array(0_st, 2_st, 1_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(misc-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(misc-misplaced-widening-cast)
              static_cast<size_t>((i1_offset + i1_step * i1) +
                                  num_pts_1 * (i3_offset + i3_step * i3) +
                                  num_pts_1 * num_pts_3 *
                                      (i2_offset + i2_step * i2));
        }
      }
    }
  } else if (neighbor_axis_permutation == make_array(2_st, 1_st, 0_st)) {
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(misc-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(misc-misplaced-widening-cast)
              static_cast<size_t>((i3_offset + i3_step * i3) +
                                  num_pts_3 * (i2_offset + i2_step * i2) +
                                  num_pts_3 * num_pts_2 *
                                      (i1_offset + i1_step * i1));
        }
      }
    }
  } else {  // make_array(1_st, 0_st, 2_st)
    for (int i3 = 0; i3 < num_pts_3; ++i3) {
      for (int i2 = 0; i2 < num_pts_2; ++i2) {
        for (int i1 = 0; i1 < num_pts_1; ++i1) {
          // NOLINTNEXTLINE(misc-misplaced-widening-cast)
          oriented_offsets[static_cast<size_t>(i1 + num_pts_1 * i2 +
                                               num_pts_1 * num_pts_2 * i3)] =
              // NOLINTNEXTLINE(misc-misplaced-widening-cast)
              static_cast<size_t>((i2_offset + i2_step * i2) +
                                  num_pts_2 * (i1_offset + i1_step * i1) +
                                  num_pts_2 * num_pts_1 *
                                      (i3_offset + i3_step * i3));
        }
      }
    }
  }
  return oriented_offsets;
}

}  // namespace

namespace OrientVariables_detail {

template <>
std::vector<size_t> oriented_offset(
    const Index<1>& extents,
    const OrientationMap<1>& orientation_of_neighbor) noexcept {
  const Direction<1> neighbor_axis =
      orientation_of_neighbor(Direction<1>::upper_xi());
  const bool is_aligned = (neighbor_axis.side() == Side::Upper);
  return compute_offset_permutation(extents, is_aligned);
}

template <>
std::vector<size_t> oriented_offset(
    const Index<2>& extents,
    const OrientationMap<2>& orientation_of_neighbor) noexcept {
  const Direction<2> neighbor_first_axis =
      orientation_of_neighbor(Direction<2>::upper_xi());
  const Direction<2> neighbor_second_axis =
      orientation_of_neighbor(Direction<2>::upper_eta());
  const bool axes_are_transposed =
      (neighbor_first_axis.dimension() > neighbor_second_axis.dimension());
  const bool neighbor_first_axis_is_aligned =
      (Side::Upper == neighbor_first_axis.side());
  const bool neighbor_second_axis_is_aligned =
      (Side::Upper == neighbor_second_axis.side());

  return compute_offset_permutation(extents, neighbor_first_axis_is_aligned,
                                    neighbor_second_axis_is_aligned,
                                    axes_are_transposed);
}

template <>
std::vector<size_t> oriented_offset(
    const Index<3>& extents,
    const OrientationMap<3>& orientation_of_neighbor) noexcept {
  const Direction<3> neighbor_first_axis =
      orientation_of_neighbor(Direction<3>::upper_xi());
  const Direction<3> neighbor_second_axis =
      orientation_of_neighbor(Direction<3>::upper_eta());
  const Direction<3> neighbor_third_axis =
      orientation_of_neighbor(Direction<3>::upper_zeta());

  const bool neighbor_first_axis_is_aligned =
      (Side::Upper == neighbor_first_axis.side());
  const bool neighbor_second_axis_is_aligned =
      (Side::Upper == neighbor_second_axis.side());
  const bool neighbor_third_axis_is_aligned =
      (Side::Upper == neighbor_third_axis.side());

  const auto neighbor_axis_permutation = make_array(
      neighbor_first_axis.dimension(), neighbor_second_axis.dimension(),
      neighbor_third_axis.dimension());

  return compute_offset_permutation(
      extents, neighbor_first_axis_is_aligned, neighbor_second_axis_is_aligned,
      neighbor_third_axis_is_aligned, neighbor_axis_permutation);
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor) noexcept {
  const Direction<2> my_slice_axis =
      (0 == sliced_dim ? Direction<2>::upper_eta() : Direction<2>::upper_xi());
  const Direction<2> neighbor_slice_axis =
      orientation_of_neighbor(my_slice_axis);
  const bool is_aligned = (neighbor_slice_axis.side() == Side::Upper);
  return compute_offset_permutation(slice_extents, is_aligned);
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<2>& slice_extents, const size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor) noexcept {
  const std::array<size_t, 2> dims_of_slice =
      (0 == sliced_dim ? make_array(1_st, 2_st)
                       : (1 == sliced_dim) ? make_array(0_st, 2_st)
                                           : make_array(0_st, 1_st));
  const bool neighbor_axes_are_transposed =
      (orientation_of_neighbor(dims_of_slice[0]) >
       orientation_of_neighbor(dims_of_slice[1]));
  const Direction<3> neighbor_first_axis =
      orientation_of_neighbor(Direction<3>(dims_of_slice[0], Side::Upper));
  const Direction<3> neighbor_second_axis =
      orientation_of_neighbor(Direction<3>(dims_of_slice[1], Side::Upper));
  const bool neighbor_first_axis_is_aligned =
      (Side::Upper == neighbor_first_axis.side());
  const bool neighbor_second_axis_is_aligned =
      (Side::Upper == neighbor_second_axis.side());

  return compute_offset_permutation(
      slice_extents, neighbor_first_axis_is_aligned,
      neighbor_second_axis_is_aligned, neighbor_axes_are_transposed);
}

}  // namespace OrientVariables_detail
