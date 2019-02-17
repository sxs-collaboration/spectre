// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_set>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_block() {
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Grid,
                                       CoordinateMaps::Identity<Dim>>));

  using coordinate_map =
      CoordinateMap<Frame::Logical, Frame::Grid, CoordinateMaps::Identity<Dim>>;
  const coordinate_map identity_map{CoordinateMaps::Identity<Dim>{}};
  Block<Dim, Frame::Grid> block(identity_map.get_clone(), 7, {});

  // Test external boundaries:
  CHECK((block.external_boundaries().size()) == 2 * Dim);

  // Test neighbors:
  CHECK((block.neighbors().size()) == 0);

  // Test id:
  CHECK((block.id()) == 7);

  // Test that the block's coordinate_map is Identity:
  const auto& map = block.coordinate_map();
  const tnsr::I<double, Dim, Frame::Logical> xi(1.0);
  const tnsr::I<double, Dim, Frame::Grid> x(1.0);
  CHECK(map(xi) == x);
  CHECK(map.inverse(x).get() == xi);

  // Test PUP
  test_serialization(block);

  // Test move semantics:
  const Block<Dim, Frame::Grid> block_copy(identity_map.get_clone(), 7, {});
  test_move_semantics(std::move(block), block_copy);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Block", "[Domain][Unit]") {
  test_block<1>();
  test_block<2>();

  // Create DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>

  // Each BlockNeighbor is an id and an OrientationMap:
  const BlockNeighbor<2> block_neighbor1(
      1, OrientationMap<2>(std::array<Direction<2>, 2>{
             {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}));
  const BlockNeighbor<2> block_neighbor2(
      2, OrientationMap<2>(std::array<Direction<2>, 2>{
             {Direction<2>::lower_xi(), Direction<2>::upper_eta()}}));
  DirectionMap<2, BlockNeighbor<2>> neighbors = {
      {Direction<2>::upper_xi(), block_neighbor1},
      {Direction<2>::lower_eta(), block_neighbor2}};
  using coordinate_map =
      CoordinateMap<Frame::Logical, Frame::Grid, CoordinateMaps::Identity<2>>;
  const coordinate_map identity_map{CoordinateMaps::Identity<2>{}};
  const Block<2, Frame::Grid> block(identity_map.get_clone(), 3,
                                    std::move(neighbors));

  // Test external boundaries:
  CHECK((block.external_boundaries().size()) == 2);

  // Test neighbors:
  CHECK((block.neighbors().size()) == 2);

  // Test id:
  CHECK((block.id()) == 3);

  // Test output:
  CHECK(get_output(block) ==
        "Block 3:\n"
        "Neighbors: "
        "([+0,Id = 1; orientation = (+0, +1)],"
        "[-1,Id = 2; orientation = (-0, +1)])\n"
        "External boundaries: (+1,-0)\n");

  // Test comparison:
  const Block<2, Frame::Grid> neighborless_block(identity_map.get_clone(), 7,
                                                 {});
  CHECK(block == block);
  CHECK(block != neighborless_block);
  CHECK(neighborless_block == neighborless_block);
}
}  // namespace domain
