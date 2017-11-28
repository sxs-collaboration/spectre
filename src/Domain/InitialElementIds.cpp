// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/InitialElementIds.hpp"

#include <iterator>

#include "Domain/ElementId.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"

template <>
std::vector<ElementId<1>> initial_element_ids<1>(
    const size_t block_id, const std::array<size_t, 1> initial_ref_levs) {
  std::vector<ElementId<1>> ids;
  ids.reserve(two_to_the(initial_ref_levs[0]));
  for (size_t x_i = 0; x_i < two_to_the(initial_ref_levs[0]); ++x_i) {
    SegmentId x_segment_id(initial_ref_levs[0], x_i);
    ids.emplace_back(block_id, make_array<1>(x_segment_id));
  }
  return ids;
}

template <>
std::vector<ElementId<2>> initial_element_ids<2>(
    const size_t block_id, const std::array<size_t, 2> initial_ref_levs) {
  std::vector<ElementId<2>> ids;
  ids.reserve(two_to_the(initial_ref_levs[0]) *
              two_to_the(initial_ref_levs[1]));
  for (size_t x_i = 0; x_i < two_to_the(initial_ref_levs[0]); ++x_i) {
    SegmentId x_segment_id(initial_ref_levs[0], x_i);
    for (size_t y_i = 0; y_i < two_to_the(initial_ref_levs[1]); ++y_i) {
      SegmentId y_segment_id(initial_ref_levs[1], y_i);
      ids.emplace_back(block_id, make_array(x_segment_id, y_segment_id));
    }
  }
  return ids;
}

template <>
std::vector<ElementId<3>> initial_element_ids<3>(
    const size_t block_id, const std::array<size_t, 3> initial_ref_levs) {
  std::vector<ElementId<3>> ids;
  ids.reserve(two_to_the(initial_ref_levs[0]) *
              two_to_the(initial_ref_levs[1]) *
              two_to_the(initial_ref_levs[2]));
  for (size_t x_i = 0; x_i < two_to_the(initial_ref_levs[0]); ++x_i) {
    SegmentId x_segment_id(initial_ref_levs[0], x_i);
    for (size_t y_i = 0; y_i < two_to_the(initial_ref_levs[1]); ++y_i) {
      SegmentId y_segment_id(initial_ref_levs[1], y_i);
      for (size_t z_i = 0; z_i < two_to_the(initial_ref_levs[2]); ++z_i) {
        SegmentId z_segment_id(initial_ref_levs[2], z_i);
        ids.emplace_back(block_id,
                         make_array(x_segment_id, y_segment_id, z_segment_id));
      }
    }
  }
  return ids;
}

template <size_t VolumeDim>
std::vector<ElementId<VolumeDim>> initial_element_ids(
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) noexcept {
  std::vector<ElementId<VolumeDim>> element_ids;
  for (size_t block_id = 0; block_id < initial_refinement_levels.size();
       ++block_id) {
    auto ids_for_block =
        initial_element_ids(block_id, initial_refinement_levels[block_id]);
    element_ids.reserve(element_ids.size() + ids_for_block.size());
    std::move(ids_for_block.begin(), ids_for_block.end(),
              std::back_inserter(element_ids));
  }
  return element_ids;
}

template std::vector<ElementId<1>> initial_element_ids<1>(
    const std::vector<std::array<size_t, 1>>&
        initial_refinement_levels) noexcept;
template std::vector<ElementId<2>> initial_element_ids<2>(
    const std::vector<std::array<size_t, 2>>&
        initial_refinement_levels) noexcept;
template std::vector<ElementId<3>> initial_element_ids<3>(
    const std::vector<std::array<size_t, 3>>&
        initial_refinement_levels) noexcept;
