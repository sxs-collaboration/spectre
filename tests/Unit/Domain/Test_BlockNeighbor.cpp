// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <string>
#include <type_traits>
#include <utility>

#include "Domain/BlockNeighbor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"  // IWYU pragma: keep
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.BlockNeighbor", "[Domain][Unit]") {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  BlockNeighbor<3> test_block_neighbor;

  // Test constructor:
  OrientationMap<3> custom_orientation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});
  BlockNeighbor<3> custom_neighbor(0, custom_orientation);
  CHECK(custom_neighbor.id() == 0);
  CHECK(custom_neighbor.orientation()(Direction<3>::upper_xi()) ==
        Direction<3>::upper_eta());
  CHECK(custom_neighbor.orientation()(Direction<3>::upper_eta()) ==
        Direction<3>::upper_zeta());
  CHECK(custom_neighbor.orientation()(Direction<3>::upper_zeta()) ==
        Direction<3>::upper_xi());

  // Test output operator:
  CHECK(get_output(custom_neighbor) == "Id = 0; orientation = (+1, +2, +0)");

  // Test comparison operator:
  CHECK(test_block_neighbor != custom_neighbor);
  CHECK(custom_neighbor == custom_neighbor);

  // Test serialization:
  test_serialization(custom_neighbor);

  // Test semantics:
  const auto custom_copy = custom_neighbor;
  test_copy_semantics(custom_neighbor);
  // clang-tidy: std::move does nothing
  test_move_semantics(std::move(custom_neighbor), custom_copy); // NOLINT
}
