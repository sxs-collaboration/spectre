// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <unordered_set>

#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Neighbors.1d", "[Domain][Unit]") {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  Neighbors<1> test_neighbors;

  // Test constructor:
  OrientationMap<1> custom_orientation(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});

  const std::unordered_set<ElementId<1>> custom_ids = []() {
    std::unordered_set<ElementId<1>> temp;
    std::array<SegmentId, 1> segment1_ids{{SegmentId(2, 3)}};
    ElementId<1> element1_id(2, segment1_ids);
    temp.insert(element1_id);
    std::array<SegmentId, 1> segment2_ids{{SegmentId(2, 2)}};
    ElementId<1> element2_id(3, segment2_ids);
    temp.insert(element2_id);
    std::array<SegmentId, 1> segment3_ids{{SegmentId(2, 1)}};
    ElementId<1> element3_id(1, segment3_ids);
    temp.insert(element3_id);
    return temp;
  }();

  Neighbors<1> custom_neighbors(custom_ids, custom_orientation);

  // Test size
  CHECK(custom_neighbors.size() == 3);

  const std::unordered_set<ElementId<1>> more_custom_ids = []() {
    std::unordered_set<ElementId<1>> temp;
    std::array<SegmentId, 1> segment4_ids{{SegmentId(2, 3)}};
    ElementId<1> element4_id(0, segment4_ids);
    temp.insert(element4_id);
    return temp;
  }();

  // Test add_ids:
  custom_neighbors.add_ids(more_custom_ids);
  CHECK(custom_neighbors.size() == 4);

  // Test set_ids_to:
  custom_neighbors.set_ids_to(custom_ids);
  CHECK(custom_neighbors.size() == 3);

  // Test serialization:
  test_serialization(custom_neighbors);

  // Test comparison:
  CHECK(test_neighbors == test_neighbors);
  CHECK(custom_neighbors == custom_neighbors);
  CHECK(test_neighbors != custom_neighbors);

  // Test iterators:
  test_iterators(custom_neighbors);
}

SPECTRE_TEST_CASE("Unit.Domain.Neighbors.2d", "[Domain][Unit]") {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  Neighbors<2> test_neighbors;

  // Test constructor:
  OrientationMap<2> custom_orientation(std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}});

  const std::unordered_set<ElementId<2>> custom_ids = []() {
    std::unordered_set<ElementId<2>> temp;
    std::array<SegmentId, 2> segment1_ids{{SegmentId(2, 3), SegmentId(1, 0)}};
    ElementId<2> element1_id(2, segment1_ids);
    temp.insert(element1_id);
    std::array<SegmentId, 2> segment2_ids{{SegmentId(2, 2), SegmentId(1, 1)}};
    ElementId<2> element2_id(3, segment2_ids);
    temp.insert(element2_id);
    std::array<SegmentId, 2> segment3_ids{{SegmentId(2, 1), SegmentId(1, 0)}};
    ElementId<2> element3_id(1, segment3_ids);
    temp.insert(element3_id);
    return temp;
  }();

  Neighbors<2> custom_neighbors(custom_ids, custom_orientation);

  // Test size
  CHECK(custom_neighbors.size() == 3);

  const std::unordered_set<ElementId<2>> more_custom_ids = []() {
    std::unordered_set<ElementId<2>> temp;
    std::array<SegmentId, 2> segment4_ids{{SegmentId(2, 3), SegmentId(1, 0)}};
    ElementId<2> element4_id(0, segment4_ids);
    temp.insert(element4_id);
    return temp;
  }();

  // Test add_ids:
  custom_neighbors.add_ids(more_custom_ids);
  CHECK(custom_neighbors.size() == 4);

  // Test set_ids_to:
  custom_neighbors.set_ids_to(custom_ids);
  CHECK(custom_neighbors.size() == 3);

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

SPECTRE_TEST_CASE("Unit.Domain.Neighbors.3d", "[Domain][Unit]") {
  // Test default constructor, only used for Charm++ serialization so no CHECK
  // calls:
  Neighbors<3> test_neighbors;

  // Test constructor:
  OrientationMap<3> custom_orientation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});

  const std::unordered_set<ElementId<3>> custom_ids = []() {
    std::unordered_set<ElementId<3>> temp;
    std::array<SegmentId, 3> segment1_ids{
        {SegmentId(2, 3), SegmentId(1, 0), SegmentId(1, 1)}};
    ElementId<3> element1_id(2, segment1_ids);
    temp.insert(element1_id);
    std::array<SegmentId, 3> segment2_ids{
        {SegmentId(2, 2), SegmentId(1, 1), SegmentId(1, 0)}};
    ElementId<3> element2_id(3, segment2_ids);
    temp.insert(element2_id);
    std::array<SegmentId, 3> segment3_ids{
        {SegmentId(2, 1), SegmentId(1, 0), SegmentId(1, 1)}};
    ElementId<3> element3_id(1, segment3_ids);
    temp.insert(element3_id);
    return temp;
  }();

  Neighbors<3> custom_neighbors(custom_ids, custom_orientation);

  // Test size
  CHECK(custom_neighbors.size() == 3);

  // Test output
  CHECK(get_output(custom_neighbors) ==
        "Ids = "
        "([B1,(L2I1,L1I0,L1I1)],[B2,(L2I3,L1I0,L1I1)],[B3,(L2I2,L1I1,L1I0)]); "
        "orientation = (+1, +2, +0)");

  // Test add_ids

  const std::unordered_set<ElementId<3>> more_custom_ids = []() {
    std::unordered_set<ElementId<3>> temp;
    std::array<SegmentId, 3> segment4_ids{
        {SegmentId(2, 3), SegmentId(1, 0), SegmentId(1, 1)}};
    ElementId<3> element4_id(0, segment4_ids);
    temp.insert(element4_id);
    return temp;
  }();

  custom_neighbors.add_ids(more_custom_ids);
  CHECK(custom_neighbors.size() == 4);

  CHECK(get_output(custom_neighbors) ==
        "Ids = "
        "([B0,(L2I3,L1I0,L1I1)],[B1,(L2I1,L1I0,L1I1)],[B2,(L2I3,L1I0,L1I1)],"
        "[B3,(L2I2,L1I1,L1I0)]); "
        "orientation = (+1, +2, +0)");

  // Test set_ids_to:
  custom_neighbors.set_ids_to(custom_ids);
  CHECK(custom_neighbors.size() == 3);

  CHECK(get_output(custom_neighbors) ==
        "Ids = "
        "([B1,(L2I1,L1I0,L1I1)],[B2,(L2I3,L1I0,L1I1)],[B3,(L2I2,L1I1,L1I0)]); "
        "orientation = (+1, +2, +0)");

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
