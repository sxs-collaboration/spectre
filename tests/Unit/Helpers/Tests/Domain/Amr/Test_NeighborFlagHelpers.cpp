// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Helpers/Domain/Amr/NeighborFlagHelpers.hpp"
#include "Helpers/Domain/Structure/NeighborHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/StdHelpers.hpp"

namespace TestHelpers::amr {
namespace {
void test_valid_neighbor_info_1d() {
  const ElementId<1> root{0};
  const ElementId<1> neighbor_root{2};
  const ElementId<1> lower_child{0, std::array{SegmentId{1, 0}}};
  const ElementId<1> upper_child{0, std::array{SegmentId{1, 1}}};
  const ElementId<1> neighbor_child{2, std::array{SegmentId{1, 0}}};
  const ElementId<1> abutting_nibling{0, std::array{SegmentId{2, 2}}};
  const auto split = std::array{::amr::Flag::Split};
  const auto join = std::array{::amr::Flag::Join};
  const auto stay = std::array{::amr::Flag::IncreaseResolution};
  const Mesh<1> mesh;
  const auto all_allowed = [&join, &stay, &split,
                            &mesh](const ElementId<1>& neighbor_id) {
    return std::vector{neighbor_info_t<1>{{neighbor_id, {join, mesh}}},
                       neighbor_info_t<1>{{neighbor_id, {stay, mesh}}},
                       neighbor_info_t<1>{{neighbor_id, {split, mesh}}}};
  };
  const auto join_not_allowed = [&stay, &split,
                                 &mesh](const ElementId<1>& neighbor_id) {
    return std::vector{neighbor_info_t<1>{{neighbor_id, {stay, mesh}}},
                       neighbor_info_t<1>{{neighbor_id, {split, mesh}}}};
  };
  const auto split_not_allowed = [&join, &stay,
                                  &mesh](const ElementId<1>& neighbor_id) {
    return std::vector{neighbor_info_t<1>{{neighbor_id, {join, mesh}}},
                       neighbor_info_t<1>{{neighbor_id, {stay, mesh}}}};
  };
  const OrientationMap<1> aligned{};
  CHECK(valid_neighbor_info(
            root, stay,
            Neighbors<1>{std::unordered_set{neighbor_root}, aligned}) ==
        join_not_allowed(neighbor_root));
  CHECK(valid_neighbor_info(
            upper_child, stay,
            Neighbors<1>{std::unordered_set{neighbor_root}, aligned}) ==
        join_not_allowed(neighbor_root));
  CHECK(valid_neighbor_info(
            lower_child, stay,
            Neighbors<1>{std::unordered_set{upper_child}, aligned}) ==
        join_not_allowed(upper_child));
  CHECK(valid_neighbor_info(
            lower_child, stay,
            Neighbors<1>{std::unordered_set{abutting_nibling}, aligned}) ==
        split_not_allowed(abutting_nibling));
  CHECK(valid_neighbor_info(
            root, stay,
            Neighbors<1>{std::unordered_set{neighbor_child}, aligned}) ==
        split_not_allowed(neighbor_child));
  CHECK(valid_neighbor_info(
            upper_child, stay,
            Neighbors<1>{std::unordered_set{neighbor_child}, aligned}) ==
        all_allowed(neighbor_child));
  CHECK(valid_neighbor_info(
            abutting_nibling, stay,
            Neighbors<1>{std::unordered_set{lower_child}, aligned}) ==
        join_not_allowed(lower_child));
}

void test_valid_neighbor_info_2d() {
  ElementId<2> element_id{0, std::array{SegmentId{2, 3}, SegmentId{3, 4}}};
  const auto join_join = std::array{::amr::Flag::Join, ::amr::Flag::Join};
  const auto join_stay =
      std::array{::amr::Flag::Join, ::amr::Flag::IncreaseResolution};
  const auto stay_join =
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Join};
  const auto stay_stay = std::array{::amr::Flag::IncreaseResolution,
                                    ::amr::Flag::IncreaseResolution};
  const auto stay_split =
      std::array{::amr::Flag::IncreaseResolution, ::amr::Flag::Split};
  const auto split_stay =
      std::array{::amr::Flag::Split, ::amr::Flag::IncreaseResolution};
  const auto split_split = std::array{::amr::Flag::Split, ::amr::Flag::Split};
  const OrientationMap<2> aligned{};
  ElementId<2> neighbor_0{1, std::array{SegmentId{2, 0}, SegmentId{3, 4}}};
  const Neighbors<2> neighbors_0{std::unordered_set{neighbor_0}, aligned};
  ::TestHelpers::domain::check_neighbors(neighbors_0, element_id,
                                         Direction<2>::upper_xi());
  const Mesh<2> mesh;
  CHECK(valid_neighbor_info(element_id, stay_stay, neighbors_0) ==
        std::vector{neighbor_info_t<2>{{neighbor_0, {join_join, mesh}}},
                    neighbor_info_t<2>{{neighbor_0, {join_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_0, {stay_join, mesh}}},
                    neighbor_info_t<2>{{neighbor_0, {stay_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_0, {stay_split, mesh}}},
                    neighbor_info_t<2>{{neighbor_0, {split_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_0, {split_split, mesh}}}});
  ElementId<2> neighbor_1{1, std::array{SegmentId{2, 0}, SegmentId{4, 8}}};
  ElementId<2> neighbor_2{1, std::array{SegmentId{2, 0}, SegmentId{4, 9}}};
  const Neighbors<2> neighbors_1_2{std::unordered_set{neighbor_1, neighbor_2},
                                   aligned};
  ::TestHelpers::domain::check_neighbors(neighbors_1_2, element_id,
                                         Direction<2>::upper_xi());
  CHECK(valid_neighbor_info(element_id, stay_stay, neighbors_1_2) ==
        std::vector{neighbor_info_t<2>{{neighbor_1, {join_join, mesh}},
                                       {neighbor_2, {join_join, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {join_stay, mesh}},
                                       {neighbor_2, {join_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {join_stay, mesh}},
                                       {neighbor_2, {stay_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {stay_join, mesh}},
                                       {neighbor_2, {stay_join, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {stay_stay, mesh}},
                                       {neighbor_2, {stay_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {stay_stay, mesh}},
                                       {neighbor_2, {split_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {split_stay, mesh}},
                                       {neighbor_2, {stay_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_1, {split_stay, mesh}},
                                       {neighbor_2, {split_stay, mesh}}}});
  ElementId<2> neighbor_3{1, std::array{SegmentId{2, 0}, SegmentId{2, 2}}};
  const Neighbors<2> neighbors_3{std::unordered_set{neighbor_3}, aligned};
  ::TestHelpers::domain::check_neighbors(neighbors_3, element_id,
                                         Direction<2>::upper_xi());
  CHECK(valid_neighbor_info(element_id, stay_stay, neighbors_3) ==
        std::vector{neighbor_info_t<2>{{neighbor_3, {join_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_3, {stay_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_3, {stay_split, mesh}}},
                    neighbor_info_t<2>{{neighbor_3, {split_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_3, {split_split, mesh}}}});
  ElementId<2> neighbor_4{1, std::array{SegmentId{2, 2}, SegmentId{2, 3}}};
  const OrientationMap<2> rotated{
      std::array{Direction<2>::lower_eta(), Direction<2>::upper_xi()}};
  const Neighbors<2> neighbors_4{std::unordered_set{neighbor_4}, rotated};
  ::TestHelpers::domain::check_neighbors(neighbors_4, element_id,
                                         Direction<2>::upper_xi());
  CHECK(valid_neighbor_info(element_id, stay_stay, neighbors_4) ==
        std::vector{neighbor_info_t<2>{{neighbor_4, {stay_join, mesh}}},
                    neighbor_info_t<2>{{neighbor_4, {stay_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_4, {stay_split, mesh}}},
                    neighbor_info_t<2>{{neighbor_4, {split_stay, mesh}}},
                    neighbor_info_t<2>{{neighbor_4, {split_split, mesh}}}});
}
}  // namespace

SPECTRE_TEST_CASE("TestHelpers.Domain.Amr.NeighborFlagHelpers",
                  "[Domain][Unit]") {
  test_valid_neighbor_info_1d();
  test_valid_neighbor_info_2d();
  // testing 3d does not test anything new
}
}  // namespace TestHelpers::amr
