// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
void test_create_initial_element(
    const ElementId<2>& element_id, const Block<2, Frame::Inertial>& block,
    const DirectionMap<2, Neighbors<2>>& expected_neighbors) noexcept {
  const auto created_element = create_initial_element(element_id, block);
  const Element<2> expected_element{element_id, expected_neighbors};
  CHECK(created_element == expected_element);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CreateInitialElement", "[Domain][Unit]") {
  OrientationMap<2> aligned(
      make_array(Direction<2>::upper_xi(), Direction<2>::upper_eta()));
  OrientationMap<2> unaligned(
      make_array(Direction<2>::lower_eta(), Direction<2>::upper_xi()));
  OrientationMap<2> inverse_of_unaligned(
      {{Direction<2>::lower_eta(), Direction<2>::upper_xi()}},
      {{Direction<2>::upper_xi(), Direction<2>::upper_eta()}});
  Block<2, Frame::Inertial> test_block(
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}),
      0,
      {{Direction<2>::upper_xi(), BlockNeighbor<2>{1, aligned}},
       {Direction<2>::upper_eta(), BlockNeighbor<2>{2, unaligned}}});

  // interior element
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 4}}}}, test_block,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 5}}}}},
                     aligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 3}}}}},
                     aligned}}});

  // element on external boundary
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{3, 0}}}}, test_block,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 0}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{3, 1}}}}},
                     aligned}}});

  // element bounding aligned neighbor block
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 4}}}}, test_block,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{1, {{SegmentId{2, 0}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 5}}}}},
                     aligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 3}}}}},
                     aligned}}});

  // element bounding unaligned neighbor block
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 7}}}}, test_block,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{2, {{SegmentId{3, 0}, SegmentId{2, 1}}}}},
                     unaligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 6}}}}},
                     aligned}}});

  // element bounding both neighbor blocks
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 7}}}}, test_block,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{1, {{SegmentId{2, 0}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{2, {{SegmentId{3, 0}, SegmentId{2, 0}}}}},
                     unaligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 6}}}}},
                     aligned}}});
}
