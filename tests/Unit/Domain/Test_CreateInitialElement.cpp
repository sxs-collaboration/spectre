// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <memory>
#include <pup.h>
#include <unordered_map>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
void test_create_initial_element(
    const domain::ElementId<2>& element_id,
    const domain::Block<2, Frame::Inertial>& block,
    const std::unordered_map<domain::Direction<2>, domain::Neighbors<2>>&
        expected_neighbors) noexcept {
  const auto created_element = create_initial_element(element_id, block);
  const domain::Element<2> expected_element{element_id, expected_neighbors};
  CHECK(created_element == expected_element);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CreateInitialElement", "[Domain][Unit]") {
  domain::OrientationMap<2> aligned(make_array(
      domain::Direction<2>::upper_xi(), domain::Direction<2>::upper_eta()));
  domain::OrientationMap<2> unaligned(make_array(
      domain::Direction<2>::lower_eta(), domain::Direction<2>::upper_xi()));
  domain::OrientationMap<2> inverse_of_unaligned(
      {{domain::Direction<2>::lower_eta(), domain::Direction<2>::upper_xi()}},
      {{domain::Direction<2>::upper_xi(), domain::Direction<2>::upper_eta()}});
  domain::Block<2, Frame::Inertial> test_block(
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}),
      0,
      {{domain::Direction<2>::upper_xi(), domain::BlockNeighbor<2>{1, aligned}},
       {domain::Direction<2>::upper_eta(),
        domain::BlockNeighbor<2>{2, unaligned}}});

  // interior element
  test_create_initial_element(
      domain::ElementId<2>{
          0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 4}}}},
      test_block,
      {{domain::Direction<2>::upper_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 4}}}}},
            aligned}},
       {domain::Direction<2>::lower_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 1}, domain::SegmentId{3, 4}}}}},
            aligned}},
       {domain::Direction<2>::upper_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 5}}}}},
            aligned}},
       {domain::Direction<2>::lower_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 3}}}}},
            aligned}}});

  // element on external boundary
  test_create_initial_element(
      domain::ElementId<2>{
          0, {{domain::SegmentId{2, 0}, domain::SegmentId{3, 0}}}},
      test_block,
      {{domain::Direction<2>::upper_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 1}, domain::SegmentId{3, 0}}}}},
            aligned}},
       {domain::Direction<2>::upper_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 0}, domain::SegmentId{3, 1}}}}},
            aligned}}});

  // element bounding aligned neighbor block
  test_create_initial_element(
      domain::ElementId<2>{
          0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 4}}}},
      test_block,
      {{domain::Direction<2>::upper_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                1, {{domain::SegmentId{2, 0}, domain::SegmentId{3, 4}}}}},
            aligned}},
       {domain::Direction<2>::lower_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 4}}}}},
            aligned}},
       {domain::Direction<2>::upper_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 5}}}}},
            aligned}},
       {domain::Direction<2>::lower_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 3}}}}},
            aligned}}});

  // element bounding unaligned neighbor block
  test_create_initial_element(
      domain::ElementId<2>{
          0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 7}}}},
      test_block,
      {{domain::Direction<2>::upper_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 7}}}}},
            aligned}},
       {domain::Direction<2>::lower_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 1}, domain::SegmentId{3, 7}}}}},
            aligned}},
       {domain::Direction<2>::upper_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                2, {{domain::SegmentId{3, 0}, domain::SegmentId{2, 1}}}}},
            unaligned}},
       {domain::Direction<2>::lower_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 6}}}}},
            aligned}}});

  // element bounding both neighbor blocks
  test_create_initial_element(
      domain::ElementId<2>{
          0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 7}}}},
      test_block,
      {{domain::Direction<2>::upper_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                1, {{domain::SegmentId{2, 0}, domain::SegmentId{3, 7}}}}},
            aligned}},
       {domain::Direction<2>::lower_xi(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 2}, domain::SegmentId{3, 7}}}}},
            aligned}},
       {domain::Direction<2>::upper_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                2, {{domain::SegmentId{3, 0}, domain::SegmentId{2, 0}}}}},
            unaligned}},
       {domain::Direction<2>::lower_eta(),
        domain::Neighbors<2>{
            {domain::ElementId<2>{
                0, {{domain::SegmentId{2, 3}, domain::SegmentId{3, 6}}}}},
            aligned}}});
}
