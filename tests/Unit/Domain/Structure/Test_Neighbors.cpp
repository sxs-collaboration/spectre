// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <unordered_set>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
void check_neighbors_1d() {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  Neighbors<1> test_neighbors;

  // Test constructor:
  OrientationMap<1> custom_orientation(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  const std::unordered_set<ElementId<1>> custom_id{
      ElementId<1>(2, {{SegmentId(2, 3)}})};
  Neighbors<1> custom_neighbors(custom_id, custom_orientation);

  // Test size
  CHECK(custom_neighbors.size() == 1);

  // Test add_ids
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([&test_neighbors, &custom_id]() {
        test_neighbors.add_ids(custom_id);
      }()),
      Catch::Matchers::ContainsSubstring("Some of the added Ids"));
#endif
  Neighbors<1> empty_neighbors{{}, {{2, custom_orientation}}, true};
  CHECK(empty_neighbors.size() == 0);
  empty_neighbors.add_ids(custom_id);
  CHECK(empty_neighbors.size() == 1);

  // Test set_ids_to:
  const std::unordered_set<ElementId<1>> other_custom_id{
      ElementId<1>(2, {{SegmentId(1, 1)}})};
  custom_neighbors.set_ids_to(other_custom_id);
  CHECK(custom_neighbors.size() == 1);
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(([&custom_neighbors]() {
                      const std::unordered_set<ElementId<1>> wrong_block_id{
                          ElementId<1>(0, {{SegmentId(2, 1)}})};
                      custom_neighbors.set_ids_to(wrong_block_id);
                    }()),
                    Catch::Matchers::ContainsSubstring("Some of the Ids"));
#endif

  // Test serialization:
  test_serialization(custom_neighbors);

  // Test comparison:
  CHECK(test_neighbors == test_neighbors);
  CHECK(custom_neighbors == custom_neighbors);
  CHECK(test_neighbors != custom_neighbors);

  // Test iterators:
  test_iterators(custom_neighbors);
}

void check_neighbors_2d() {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  Neighbors<2> test_neighbors;

  // Test constructor:
  OrientationMap<2> custom_orientation(std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}});
  const std::unordered_set<ElementId<2>> custom_id{
      ElementId<2>(2, {{SegmentId(2, 3), SegmentId(1, 0)}})};
  Neighbors<2> custom_neighbors(custom_id, custom_orientation);

  // Test size
  CHECK(custom_neighbors.size() == 1);

  // Test add_ids:
  const std::unordered_set<ElementId<2>> other_custom_id{
      ElementId<2>(2, {{SegmentId(2, 2), SegmentId(1, 0)}})};
  custom_neighbors.add_ids(other_custom_id);
  CHECK(custom_neighbors.size() == 2);

  // Test set_ids_to:
  custom_neighbors.set_ids_to(custom_id);
  CHECK(custom_neighbors.size() == 1);

  // Test serialization:
  test_serialization(custom_neighbors);

  // Test comparison:
  CHECK(test_neighbors == test_neighbors);
  CHECK(custom_neighbors == custom_neighbors);
  CHECK(test_neighbors != custom_neighbors);

  // Test iterators:
  test_iterators(custom_neighbors);

  // Test semantics:
  const auto custom_copy = custom_neighbors;
  test_copy_semantics(test_neighbors);
  test_move_semantics(std::move(custom_neighbors), custom_copy);
}

void check_neighbors_3d() {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  Neighbors<3> test_neighbors;

  // Test constructor:
  OrientationMap<3> custom_orientation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});
  const std::unordered_set<ElementId<3>> custom_ids{
      ElementId<3>(2, {{SegmentId(2, 3), SegmentId(1, 0), SegmentId(1, 1)}}),
      ElementId<3>(2, {{SegmentId(2, 2), SegmentId(1, 1), SegmentId(1, 0)}}),
      ElementId<3>(2, {{SegmentId(2, 1), SegmentId(1, 0), SegmentId(1, 1)}})};
  Neighbors<3> custom_neighbors(custom_ids, custom_orientation);

  // Test size
  CHECK(custom_neighbors.size() == 3);

  // Test output
  CHECK(get_output(custom_neighbors) ==
        "Ids = "
        "([B2,(L2I1,L1I0,L1I1)],[B2,(L2I2,L1I1,L1I0)],[B2,(L2I3,L1I0,L1I1)]); "
        "orientations = ([2,(+1, +2, +0)]); conforming = true");

  // Test add_ids
  const std::unordered_set<ElementId<3>> other_custom_id{
      ElementId<3>(2, {{SegmentId(2, 3), SegmentId(1, 1), SegmentId(1, 1)}})};
  custom_neighbors.add_ids(other_custom_id);
  CHECK(custom_neighbors.size() == 4);

  CHECK(get_output(custom_neighbors) ==
        "Ids = "
        "([B2,(L2I1,L1I0,L1I1)],[B2,(L2I2,L1I1,L1I0)],[B2,(L2I3,L1I0,L1I1)],["
        "B2,(L2I3,L1I1,L1I1)]); "
        "orientations = ([2,(+1, +2, +0)]); conforming = true");

  // Test set_ids_to:
  custom_neighbors.set_ids_to(custom_ids);
  CHECK(custom_neighbors.size() == 3);

  CHECK(get_output(custom_neighbors) ==
        "Ids = "
        "([B2,(L2I1,L1I0,L1I1)],[B2,(L2I2,L1I1,L1I0)],[B2,(L2I3,L1I0,L1I1)]); "
        "orientations = ([2,(+1, +2, +0)]); conforming = true");

  // Test serialization:
  test_serialization(custom_neighbors);

  // Test comparison:
  CHECK(test_neighbors == test_neighbors);
  CHECK(custom_neighbors == custom_neighbors);
  CHECK(test_neighbors != custom_neighbors);

  // Test iterators:
  test_iterators(custom_neighbors);

  // Test semantics:
  const auto custom_copy = custom_neighbors;
  test_copy_semantics(test_neighbors);
  test_move_semantics(std::move(custom_neighbors), custom_copy);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.Neighbors", "[Domain][Unit]") {
  check_neighbors_1d();
  check_neighbors_2d();
  check_neighbors_3d();
}
