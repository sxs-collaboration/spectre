// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <vector>

#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/InitialElementIds.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"

namespace {
template <size_t VolumeDim>
void test_initial_element_ids(
    const std::vector<ElementId<VolumeDim>>& element_ids,
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) noexcept {
  size_t expected_number_of_elements = 0;
  for (const auto& initial_refinement_levels_of_block :
       initial_refinement_levels) {
    size_t expected_number_of_elements_in_block = 1;
    for (size_t d = 0; d < VolumeDim; ++d) {
      expected_number_of_elements_in_block *=
          two_to_the(gsl::at(initial_refinement_levels_of_block, d));
    }
    expected_number_of_elements += expected_number_of_elements_in_block;
  }
  CHECK(expected_number_of_elements == element_ids.size());
  const boost::rational<size_t> expected_logical_volume_of_blocks(
      initial_refinement_levels.size());
  boost::rational<size_t> logical_volume_of_blocks(0);
  for (const auto& element_id : element_ids) {
    logical_volume_of_blocks += fraction_of_block_volume(element_id);
  }
  CHECK(expected_logical_volume_of_blocks == logical_volume_of_blocks);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.InitialElementIds", "[Domain][Unit]") {
  const std::vector<std::array<size_t, 1>> initial_refinement_levels_1d{{{2}},
                                                                        {{3}}};
  const auto element_ids_1d = initial_element_ids(initial_refinement_levels_1d);
  test_initial_element_ids(element_ids_1d, initial_refinement_levels_1d);

  const std::vector<std::array<size_t, 2>> initial_refinement_levels_2d{
      {{2, 0}}, {{3, 1}}};
  const auto element_ids_2d = initial_element_ids(initial_refinement_levels_2d);
  test_initial_element_ids(element_ids_2d, initial_refinement_levels_2d);

  const std::vector<std::array<size_t, 3>> initial_refinement_levels_3d{
      {{4, 2, 1}}, {{0, 3, 2}}};
  const auto element_ids_3d = initial_element_ids(initial_refinement_levels_3d);
  test_initial_element_ids(element_ids_3d, initial_refinement_levels_3d);
}
