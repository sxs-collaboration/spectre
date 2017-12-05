// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/Direction.hpp"
#include "Domain/DomainHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

// Tests that get_common_global_corners is an order-sensitive intersection.
SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.Boundaries", "[Domain][Unit]") {
  std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<3>(corners_of_all_blocks, &neighbors_of_all_blocks);

  std::array<Direction<3>, 3> mapped_directions{{Direction<3>::upper_xi(),
                                                 Direction<3>::upper_eta(),
                                                 Direction<3>::upper_zeta()}};
  OrientationMap<3> expected_orientation(mapped_directions);
  CHECK(
      (neighbors_of_all_blocks)[0][Direction<3>::lower_zeta()].orientation() ==
      expected_orientation);

  const PairOfFaces x_faces{{1, 3, 5, 7}, {0, 2, 4, 6}};

  std::vector<PairOfFaces> identifications{x_faces};
  set_periodic_boundaries<3>(identifications, corners_of_all_blocks,
                             &neighbors_of_all_blocks);
  OrientationMap<3> expected_identification(mapped_directions);
  CHECK((neighbors_of_all_blocks)[0][Direction<3>::upper_xi()].orientation() ==
        expected_identification);
}
