// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <unordered_set>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
template <size_t VolumeDim>
void check_element_work(const typename Element<VolumeDim>::Neighbors_t&
                            neighbors_in_largest_dimension,
                        const size_t expected_number_of_neighbors) {
  const ElementId<VolumeDim> id{5};
  const Element<VolumeDim> element(id, neighbors_in_largest_dimension);

  CHECK(element.id() == id);
  CHECK(element.neighbors() == neighbors_in_largest_dimension);
  CHECK(element.number_of_neighbors() == expected_number_of_neighbors);
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    // The highest spatial dimension has neighbors; else, external boundary.
    if (direction.dimension() == VolumeDim - 1) {
      CHECK_FALSE(element.external_boundaries().contains(direction));
      CHECK(element.internal_boundaries().contains(direction));
      CHECK(element.neighbors().contains(direction));
      CHECK(element.face_types().at(direction) ==
            domain::FaceType::ConformingAligned);
    } else {
      CHECK(element.external_boundaries().contains(direction));
      CHECK_FALSE(element.internal_boundaries().contains(direction));
      CHECK_FALSE(element.neighbors().contains(direction));
      CHECK(element.face_types().at(direction) == domain::FaceType::External);
    }
    // All faces of a hypercube element are boundaries (no topological faces).
    CHECK(element.all_boundaries().contains(direction));
  }
  CHECK(element.neighbors().size() == element.internal_boundaries().size());
  CHECK(element.internal_boundaries().size() +
            element.external_boundaries().size() ==
        2 * VolumeDim);
  CHECK(element.all_boundaries().size() == 2 * VolumeDim);
  CHECK(element == element);
  CHECK_FALSE(element != element);

  const Element<VolumeDim> element_diff_id(ElementId<VolumeDim>(3),
                                           neighbors_in_largest_dimension);
  CHECK(element != element_diff_id);
  CHECK_FALSE(element == element_diff_id);

  const Element<VolumeDim> element_diff_neighbors(
      id, typename Element<VolumeDim>::Neighbors_t{
              {Direction<VolumeDim>::lower_xi(),
               neighbors_in_largest_dimension.at(
                   Direction<VolumeDim>(VolumeDim - 1, Side::Upper))}});
  CHECK(element != element_diff_neighbors);
  CHECK_FALSE(element == element_diff_neighbors);

  CHECK(get_output(element) == "Element " + get_output(element.id()) +
                                   ":\n"
                                   "  Topology: " +
                                   get_output(element.topologies()) +
                                   "\n"
                                   "  Neighbors: " +
                                   get_output(element.neighbors()) +
                                   "\n"
                                   "  External boundaries: " +
                                   get_output(element.external_boundaries()) +
                                   "\n"
                                   "  Face types: " +
                                   get_output(element.face_types()) + "\n");

  test_serialization(element);
}

void check_element_1d() {
  const Neighbors<1> neighbors_lower_xi(
      std::unordered_set<ElementId<1>>{ElementId<1>(7)},
      OrientationMap<1>::create_aligned());
  const Neighbors<1> neighbors_upper_xi(
      std::unordered_set<ElementId<1>>{ElementId<1>(7)},
      OrientationMap<1>::create_aligned());
  const typename Element<1>::Neighbors_t xi_neighbors{
      {Direction<1>::lower_xi(), neighbors_lower_xi},
      {Direction<1>::upper_xi(), neighbors_upper_xi}};
  check_element_work<1>(xi_neighbors, 2);
}

void check_element_2d() {
  const SegmentId root_segment{0, 0};
  const Neighbors<2> neighbors_lower_eta(
      std::unordered_set<ElementId<2>>{
          ElementId<2>(7,
                       {{root_segment.id_of_child(Side::Lower), root_segment}}),
          ElementId<2>(
              7, {{root_segment.id_of_child(Side::Upper), root_segment}})},
      OrientationMap<2>::create_aligned());
  const Neighbors<2> neighbors_upper_eta(
      std::unordered_set<ElementId<2>>{
          ElementId<2>(8,
                       {{root_segment.id_of_child(Side::Lower), root_segment}}),
          ElementId<2>(
              8, {{root_segment.id_of_child(Side::Upper), root_segment}})},
      OrientationMap<2>::create_aligned());
  const typename Element<2>::Neighbors_t eta_neighbors{
      {Direction<2>::lower_eta(), neighbors_lower_eta},
      {Direction<2>::upper_eta(), neighbors_upper_eta}};
  check_element_work<2>(eta_neighbors, 4);
}

void check_element_3d() {
  const SegmentId root_segment{0, 0};
  const Neighbors<3> neighbors_lower_zeta(
      std::unordered_set<ElementId<3>>{
          ElementId<3>(7,
                       {{root_segment.id_of_child(Side::Lower),
                         root_segment.id_of_child(Side::Lower), root_segment}}),
          ElementId<3>(7,
                       {{root_segment.id_of_child(Side::Lower),
                         root_segment.id_of_child(Side::Upper), root_segment}}),
          ElementId<3>(7,
                       {{root_segment.id_of_child(Side::Upper),
                         root_segment.id_of_child(Side::Lower), root_segment}}),
          ElementId<3>(
              7, {{root_segment.id_of_child(Side::Upper),
                   root_segment.id_of_child(Side::Upper), root_segment}})},
      OrientationMap<3>::create_aligned());
  const Neighbors<3> neighbors_upper_zeta(
      std::unordered_set<ElementId<3>>{
          ElementId<3>(8,
                       {{root_segment.id_of_child(Side::Lower),
                         root_segment.id_of_child(Side::Lower), root_segment}}),
          ElementId<3>(8,
                       {{root_segment.id_of_child(Side::Lower),
                         root_segment.id_of_child(Side::Upper), root_segment}}),
          ElementId<3>(8,
                       {{root_segment.id_of_child(Side::Upper),
                         root_segment.id_of_child(Side::Lower), root_segment}}),
          ElementId<3>(
              8, {{root_segment.id_of_child(Side::Upper),
                   root_segment.id_of_child(Side::Upper), root_segment}})},
      OrientationMap<3>::create_aligned());
  const typename Element<3>::Neighbors_t zeta_neighbors{
      {Direction<3>::lower_zeta(), neighbors_lower_zeta},
      {Direction<3>::upper_zeta(), neighbors_upper_zeta}};
  check_element_work<3>(zeta_neighbors, 8);
}

void check_spherical_shell() {
  const Element<3> spherical_shell(
      ElementId<3>{5}, DirectionMap<3, Neighbors<3>>{},
      std::array{domain::Topology::I1, domain::Topology::S2Colatitude,
                 domain::Topology::S2Longitude});
  CHECK(spherical_shell.external_boundaries().size() == 2);
  CHECK(
      spherical_shell.external_boundaries().contains(Direction<3>::lower_xi()));
  CHECK(
      spherical_shell.external_boundaries().contains(Direction<3>::upper_xi()));
  CHECK(spherical_shell.neighbors().empty());
  // Only the two radial faces are boundaries; the angular faces are
  // topological.
  CHECK(spherical_shell.all_boundaries().size() == 2);
  CHECK(spherical_shell.all_boundaries().contains(Direction<3>::lower_xi()));
  CHECK(spherical_shell.all_boundaries().contains(Direction<3>::upper_xi()));
  for (const auto& direction : Direction<3>::all_directions()) {
    if (direction.dimension() == 0) {
      CHECK(spherical_shell.external_boundaries().contains(direction));
      CHECK_FALSE(spherical_shell.internal_boundaries().contains(direction));
      CHECK(spherical_shell.all_boundaries().contains(direction));
      CHECK_FALSE(spherical_shell.neighbors().contains(direction));
      CHECK(spherical_shell.face_types().at(direction) ==
            domain::FaceType::External);
    } else {
      CHECK_FALSE(spherical_shell.external_boundaries().contains(direction));
      CHECK_FALSE(spherical_shell.internal_boundaries().contains(direction));
      CHECK_FALSE(spherical_shell.all_boundaries().contains(direction));
      CHECK_FALSE(spherical_shell.neighbors().contains(direction));
      CHECK(spherical_shell.face_types().at(direction) ==
            domain::FaceType::Topological);
    }
  }
}

void check_assert() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const Neighbors<1> element_neighbors(
            std::unordered_set<ElementId<1>>{ElementId<1>{3}},
            OrientationMap<1>::create_aligned());
        const DirectionMap<1, Neighbors<1>> neighbors{
            {Direction<1>::lower_xi(), element_neighbors}};
        const Element<1> loop(ElementId<1>{2}, neighbors,
                              std::array{domain::Topology::S1});
      }()),
      Catch::Matchers::ContainsSubstring(
          "Cannot specify a neighbor in a direction with no boundary"));
#endif
}

void test_nonconforming_blocks() {
  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  std::vector<Block<2>> blocks;
  blocks.emplace_back(
      nullptr, 0,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::upper_xi(),
           BlockNeighbors<2>{
               {1, 2, 3, 4},
               {{1, aligned}, {2, aligned}, {3, aligned}, {4, aligned}},
               false}}},
      "Annulus", std::array{domain::Topology::I1, domain::Topology::S1});
  blocks.emplace_back(
      nullptr, 1,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(),
           BlockNeighbors<2>{{0}, {{0, aligned}}, false}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{2, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{4, aligned}}},
      "North", std::array{domain::Topology::I1, domain::Topology::I1});
  blocks.emplace_back(
      nullptr, 2,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(),
           BlockNeighbors<2>{{0}, {{0, aligned}}, false}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{3, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{1, aligned}}},
      "East", std::array{domain::Topology::I1, domain::Topology::I1});
  blocks.emplace_back(
      nullptr, 3,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(),
           BlockNeighbors<2>{{0}, {{0, aligned}}, false}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{4, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{2, aligned}}},
      "South", std::array{domain::Topology::I1, domain::Topology::I1});
  blocks.emplace_back(
      nullptr, 4,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(),
           BlockNeighbors<2>{{0}, {{0, aligned}}, false}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{1, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{3, aligned}}},
      "West", std::array{domain::Topology::I1, domain::Topology::I1});
  const std::vector<std::array<size_t, 2>> initial_refinement_levels{
      std::array{2_st, 0_st}, std::array{0_st, 1_st}, std::array{0_st, 1_st},
      std::array{0_st, 1_st}, std::array{0_st, 1_st}};
  const ElementId<2> annulus_id{0,
                                std::array{SegmentId{2, 3}, SegmentId{0, 0}}};
  const ElementId<2> wedge_id{1, std::array{SegmentId{0, 0}, SegmentId{1, 0}}};
  const auto annulus = domain::create_initial_element(
      annulus_id, blocks, initial_refinement_levels);
  CHECK(annulus.number_of_neighbors() == 9);
  CHECK_FALSE(annulus.external_boundaries().contains(Direction<2>::lower_xi()));
  CHECK(annulus.internal_boundaries().contains(Direction<2>::lower_xi()));
  CHECK(annulus.neighbors().contains(Direction<2>::lower_xi()));
  CHECK(annulus.face_types().at(Direction<2>::lower_xi()) ==
        domain::FaceType::ConformingAligned);
  CHECK_FALSE(annulus.external_boundaries().contains(Direction<2>::upper_xi()));
  CHECK(annulus.internal_boundaries().contains(Direction<2>::upper_xi()));
  CHECK(annulus.neighbors().contains(Direction<2>::upper_xi()));
  CHECK(annulus.face_types().at(Direction<2>::upper_xi()) ==
        domain::FaceType::MultipleNonconforming);
  CHECK_FALSE(
      annulus.external_boundaries().contains(Direction<2>::lower_eta()));
  CHECK_FALSE(
      annulus.internal_boundaries().contains(Direction<2>::lower_eta()));
  CHECK_FALSE(annulus.neighbors().contains(Direction<2>::lower_eta()));
  CHECK(annulus.face_types().at(Direction<2>::lower_eta()) ==
        domain::FaceType::Topological);
  CHECK_FALSE(
      annulus.external_boundaries().contains(Direction<2>::upper_eta()));
  CHECK_FALSE(
      annulus.internal_boundaries().contains(Direction<2>::upper_eta()));
  CHECK_FALSE(annulus.neighbors().contains(Direction<2>::upper_eta()));
  CHECK(annulus.face_types().at(Direction<2>::upper_eta()) ==
        domain::FaceType::Topological);

  const auto wedge = domain::create_initial_element(wedge_id, blocks,
                                                    initial_refinement_levels);
  CHECK(wedge.number_of_neighbors() == 3);
  CHECK_FALSE(wedge.external_boundaries().contains(Direction<2>::lower_xi()));
  CHECK(wedge.internal_boundaries().contains(Direction<2>::lower_xi()));
  CHECK(wedge.neighbors().contains(Direction<2>::lower_xi()));
  CHECK(wedge.face_types().at(Direction<2>::lower_xi()) ==
        domain::FaceType::SingleNonconforming);
  CHECK(wedge.external_boundaries().contains(Direction<2>::upper_xi()));
  CHECK_FALSE(wedge.internal_boundaries().contains(Direction<2>::upper_xi()));
  CHECK_FALSE(wedge.neighbors().contains(Direction<2>::upper_xi()));
  CHECK(wedge.face_types().at(Direction<2>::upper_xi()) ==
        domain::FaceType::External);
  CHECK_FALSE(wedge.external_boundaries().contains(Direction<2>::lower_eta()));
  CHECK(wedge.internal_boundaries().contains(Direction<2>::lower_eta()));
  CHECK(wedge.neighbors().contains(Direction<2>::lower_eta()));
  CHECK(wedge.face_types().at(Direction<2>::lower_eta()) ==
        domain::FaceType::ConformingAligned);
  CHECK_FALSE(wedge.external_boundaries().contains(Direction<2>::upper_eta()));
  CHECK(wedge.internal_boundaries().contains(Direction<2>::upper_eta()));
  CHECK(wedge.neighbors().contains(Direction<2>::upper_eta()));
  CHECK(wedge.face_types().at(Direction<2>::upper_eta()) ==
        domain::FaceType::ConformingAligned);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.Element", "[Domain][Unit]") {
  check_element_1d();
  check_element_2d();
  check_element_3d();
  check_spherical_shell();
  check_assert();
  test_nonconforming_blocks();
}
