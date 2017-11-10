// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/InitialElementIds.hpp"

#include "Domain/ElementId.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"

template <>
std::vector<ElementId<1>> initial_element_ids(
    const std::vector<std::array<size_t, 1>>&
        initial_refinement_levels) noexcept {
  std::vector<ElementId<1>> element_ids;
  for (size_t block_id = 0; block_id < initial_refinement_levels.size();
       ++block_id) {
    const size_t xi_refinement_level = initial_refinement_levels[block_id][0];
    for (size_t xi_index = 0; xi_index < two_to_the(xi_refinement_level);
         ++xi_index) {
      element_ids.emplace_back(
          block_id, make_array<1>(SegmentId(xi_refinement_level, xi_index)));
    }
  }
  return element_ids;
}

template <>
std::vector<ElementId<2>> initial_element_ids(
    const std::vector<std::array<size_t, 2>>&
        initial_refinement_levels) noexcept {
  std::vector<ElementId<2>> element_ids;
  for (size_t block_id = 0; block_id < initial_refinement_levels.size();
       ++block_id) {
    const size_t xi_refinement_level = initial_refinement_levels[block_id][0];
    for (size_t xi_index = 0; xi_index < two_to_the(xi_refinement_level);
         ++xi_index) {
      const size_t eta_refinement_level =
          initial_refinement_levels[block_id][1];
      for (size_t eta_index = 0; eta_index < two_to_the(eta_refinement_level);
           ++eta_index) {
        element_ids.emplace_back(
            block_id, make_array(SegmentId(xi_refinement_level, xi_index),
                                 SegmentId(eta_refinement_level, eta_index)));
      }
    }
  }
  return element_ids;
}

template <>
std::vector<ElementId<3>> initial_element_ids(
    const std::vector<std::array<size_t, 3>>&
        initial_refinement_levels) noexcept {
  std::vector<ElementId<3>> element_ids;
  for (size_t block_id = 0; block_id < initial_refinement_levels.size();
       ++block_id) {
    const size_t xi_refinement_level = initial_refinement_levels[block_id][0];
    for (size_t xi_index = 0; xi_index < two_to_the(xi_refinement_level);
         ++xi_index) {
      const size_t eta_refinement_level =
          initial_refinement_levels[block_id][1];
      for (size_t eta_index = 0; eta_index < two_to_the(eta_refinement_level);
           ++eta_index) {
        const size_t zeta_refinement_level =
            initial_refinement_levels[block_id][2];
        for (size_t zeta_index = 0;
             zeta_index < two_to_the(zeta_refinement_level); ++zeta_index) {
          element_ids.emplace_back(
              block_id,
              make_array(SegmentId(xi_refinement_level, xi_index),
                         SegmentId(eta_refinement_level, eta_index),
                         SegmentId(zeta_refinement_level, zeta_index)));
        }
      }
    }
  }
  return element_ids;
}
