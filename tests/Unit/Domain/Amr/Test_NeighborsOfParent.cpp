// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/NeighborsOfParent.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
SegmentId s_00{0, 0};
SegmentId s_10{1, 0};
SegmentId s_11{1, 1};

void test_periodic_interval() {
  OrientationMap<1> aligned{};
  ElementId<1> parent_id{0, std::array{s_00}};
  ElementId<1> child_1_id{0, std::array{s_10}};
  ElementId<1> child_2_id{0, std::array{s_11}};

  DirectionMap<1, Neighbors<1>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{child_2_id}, aligned});
  child_1_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{child_2_id}, aligned});
  Element<1> child_1{child_1_id, std::move(child_1_neighbors)};
  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      child_1_neighbor_flags{{child_2_id, {{amr::Flag::Join}}}};

  DirectionMap<1, Neighbors<1>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{child_1_id}, aligned});
  Element<1> child_2{child_2_id, std::move(child_2_neighbors)};
  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      child_2_neighbor_flags{{child_1_id, {{amr::Flag::Join}}}};

  std::vector<std::tuple<
      const Element<1>&,
      const std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>&>>
      children_elements_and_neighbor_flags;
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_1, child_1_neighbor_flags));
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_2, child_2_neighbor_flags));

  const auto parent_neighbors =
      amr::neighbors_of_parent(parent_id, children_elements_and_neighbor_flags);
  DirectionMap<1, Neighbors<1>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{parent_id}, aligned});
  expected_parent_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{parent_id}, aligned});
  CHECK(parent_neighbors == expected_parent_neighbors);
}

void test_interval() {
  OrientationMap<1> aligned{};
  OrientationMap<1> flipped{std::array{Direction<1>::lower_xi()}};
  ElementId<1> parent_id{0, std::array{s_00}};
  ElementId<1> child_1_id{0, std::array{s_10}};
  ElementId<1> child_2_id{0, std::array{s_11}};
  ElementId<1> lower_neighbor_id{1, std::array{s_11}};
  ElementId<1> upper_neighbor_id{2, std::array{s_00}};

  DirectionMap<1, Neighbors<1>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{lower_neighbor_id}, aligned});
  child_1_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{child_2_id}, aligned});
  Element<1> child_1{child_1_id, std::move(child_1_neighbors)};
  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      child_1_neighbor_flags{{lower_neighbor_id, {{amr::Flag::DoNothing}}},
                             {child_2_id, {{amr::Flag::Join}}}};

  DirectionMap<1, Neighbors<1>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{upper_neighbor_id}, flipped});
  Element<1> child_2{child_2_id, std::move(child_2_neighbors)};
  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      child_2_neighbor_flags{{child_1_id, {{amr::Flag::Join}}},
                             {upper_neighbor_id, {{amr::Flag::Split}}}};

  std::vector<std::tuple<
      const Element<1>&,
      const std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>&>>
      children_elements_and_neighbor_flags;
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_1, child_1_neighbor_flags));
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_2, child_2_neighbor_flags));

  const auto parent_neighbors =
      amr::neighbors_of_parent(parent_id, children_elements_and_neighbor_flags);
  DirectionMap<1, Neighbors<1>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{lower_neighbor_id}, aligned});
  ElementId<1> split_upper_neighbor_id{2, std::array{s_11}};
  expected_parent_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{split_upper_neighbor_id}, flipped});
  CHECK(parent_neighbors == expected_parent_neighbors);
}

void test_rectangle() {
  OrientationMap<2> aligned{};
  OrientationMap<2> rotated{
      std::array{Direction<2>::lower_eta(), Direction<2>::upper_xi()}};

  ElementId<2> parent_id{0, std::array{s_00, s_00}};
  ElementId<2> child_1_id{0, std::array{s_10, s_10}};
  ElementId<2> child_2_id{0, std::array{s_11, s_10}};
  ElementId<2> child_3_id{0, std::array{s_10, s_11}};
  ElementId<2> child_4_id{0, std::array{s_11, s_11}};
  ElementId<2> neighbor_1_id{1, std::array{s_11, s_00}};
  ElementId<2> neighbor_2_id{2, std::array{s_10, s_00}};
  ElementId<2> neighbor_3_id{2, std::array{s_11, s_11}};

  std::array join_join{amr::Flag::Join, amr::Flag::Join};
  std::array neighbor_1_flags{amr::Flag::Join, amr::Flag::DoNothing};
  std::array neighbor_2_flags{amr::Flag::DoNothing, amr::Flag::Split};
  std::array neighbor_3_flags{amr::Flag::DoNothing, amr::Flag::Join};

  ElementId<2> neighbor_4_id{1, std::array{s_00, s_00}};
  ElementId<2> neighbor_5_id{2, std::array{s_10, s_11}};
  ElementId<2> neighbor_6_id{2, std::array{s_11, s_00}};

  DirectionMap<2, Neighbors<2>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{neighbor_1_id}, aligned});
  child_1_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{child_2_id}, aligned});
  child_1_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_3_id}, aligned});
  child_1_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_3_id}, aligned});
  Element<2> child_1{child_1_id, std::move(child_1_neighbors)};
  std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>
      child_1_neighbor_flags{{neighbor_1_id, neighbor_1_flags},
                             {child_2_id, join_join},
                             {child_3_id, join_join}};

  DirectionMap<2, Neighbors<2>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{neighbor_2_id}, rotated});
  child_2_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_4_id}, aligned});
  child_2_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_4_id}, aligned});
  Element<2> child_2{child_2_id, std::move(child_2_neighbors)};
  std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>
      child_2_neighbor_flags{{child_1_id, join_join},
                             {child_4_id, join_join},
                             {neighbor_2_id, neighbor_2_flags}};

  DirectionMap<2, Neighbors<2>> child_3_neighbors{};
  child_3_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{neighbor_1_id}, aligned});
  child_3_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{child_4_id}, aligned});
  child_3_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_1_id}, aligned});
  child_3_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_1_id}, aligned});
  Element<2> child_3{child_3_id, std::move(child_3_neighbors)};
  std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>
      child_3_neighbor_flags{{neighbor_1_id, neighbor_1_flags},
                             {child_1_id, join_join},
                             {child_4_id, join_join}};

  DirectionMap<2, Neighbors<2>> child_4_neighbors{};
  child_4_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{child_3_id}, aligned});
  child_4_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{neighbor_3_id}, rotated});
  child_4_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_2_id}, aligned});
  child_4_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_2_id}, aligned});
  Element<2> child_4{child_4_id, std::move(child_4_neighbors)};
  std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>
      child_4_neighbor_flags{{child_2_id, join_join},
                             {child_3_id, join_join},
                             {neighbor_3_id, neighbor_3_flags}};

  std::vector<std::tuple<
      const Element<2>&,
      const std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>&>>
      children_elements_and_neighbor_flags;
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_1, child_1_neighbor_flags));
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_2, child_2_neighbor_flags));
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_3, child_3_neighbor_flags));
  children_elements_and_neighbor_flags.emplace_back(
      std::forward_as_tuple(child_4, child_4_neighbor_flags));

  const auto parent_neighbors =
      amr::neighbors_of_parent(parent_id, children_elements_and_neighbor_flags);
  DirectionMap<2, Neighbors<2>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{neighbor_4_id}, aligned});
  expected_parent_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{neighbor_5_id, neighbor_6_id}, rotated});
  expected_parent_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{parent_id}, aligned});
  expected_parent_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{parent_id}, aligned});
  CHECK(parent_neighbors == expected_parent_neighbors);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.NeighborsOfParent", "[Domain][Unit]") {
  test_periodic_interval();
  test_interval();
  test_rectangle();
}
