// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/InitializeParent.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<volume_dim>,
                 domain::Tags::Mesh<volume_dim>, amr::Tags::Flags<volume_dim>,
                 amr::Tags::NeighborFlags<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<Component<Metavariables>>;
};

// Test setup showing xi and eta dimensions which are hp-refined; only
// p-refinement in zeta dimension

// Before refinement:                     After refinement:
//
//       |-----------|                         |-----------|
//       |           |                         |           |
//       |    N5     |                         |    N5     |
//       |           |                         |           |
// |-----|-----------|-----------|       |-----|-----------|-----------|
// |     |           |           |       |     |           |           |
// |  N6 |    C3     |           |       |  N6 |           |    N11    |
// |     |           |           |       |     |           |           |
// |--|--|-----|-----|    N4     |       |-----|     P     |-----------|
//    |N7|     |     |           |       |     |           |           |
//    |--|  C1 |  C2 |           |       | N12 |           |    N10    |
//    |N8|     |     |           |       |     |           |           |
//    |--|--|--|-----|-----------|       |-----|--|--|-----|-----------|
//       |  |  |     |                         |     |     |
//       |N1|N2|  N3 |                         |  N9 |  N3 |
//       |  |  |     |                         |     |     |
//       |-----|-----|                         |-----|-----|
//
// Block setup is:
// Elements C1, C2, and C3 are in Block 0
// Element N4 is in Block 1 which is aligned with Block 0
// Element N5 is in Block 2 which is rotated by 90 degrees counter clockwise
// Elments N6, N7, and N8 are in Block 3 which is anti-aligned with Block 0
// Elements N1, N2, and N3 are in Block 4 which is rotated 90 deg. clockwise
void test() {
  using my_component = Component<Metavariables>;

  OrientationMap<3> aligned{};
  OrientationMap<3> b1_orientation{};
  OrientationMap<3> b2_orientation{std::array{Direction<3>::lower_eta(),
                                              Direction<3>::upper_xi(),
                                              Direction<3>::upper_zeta()}};
  OrientationMap<3> b3_orientation{std::array{Direction<3>::lower_xi(),
                                              Direction<3>::lower_eta(),
                                              Direction<3>::upper_zeta()}};
  OrientationMap<3> b4_orientation{std::array{Direction<3>::upper_eta(),
                                              Direction<3>::lower_xi(),
                                              Direction<3>::upper_zeta()}};

  SegmentId s_00{0, 0};
  SegmentId s_10{1, 0};
  SegmentId s_11{1, 1};
  SegmentId s_20{2, 0};
  SegmentId s_21{2, 1};
  SegmentId s_22{2, 2};
  SegmentId s_23{2, 3};

  const ElementId<3> parent_id{0, std::array{s_00, s_00, s_00}};

  const ElementId<3> child_1_id{0, std::array{s_10, s_10, s_00}};
  const ElementId<3> child_2_id{0, std::array{s_11, s_10, s_00}};
  const ElementId<3> child_3_id{0, std::array{s_00, s_11, s_00}};

  const ElementId<3> neighbor_1_id{4, std::array{s_10, s_20, s_00}};
  const ElementId<3> neighbor_2_id{4, std::array{s_10, s_21, s_00}};
  const ElementId<3> neighbor_3_id{4, std::array{s_10, s_11, s_00}};
  const ElementId<3> neighbor_9_id{4, std::array{s_10, s_10, s_00}};

  const ElementId<3> neighbor_4_id{1, std::array{s_00, s_00, s_00}};
  const ElementId<3> neighbor_10_id{1, std::array{s_00, s_10, s_00}};
  const ElementId<3> neighbor_11_id{1, std::array{s_00, s_11, s_00}};

  const ElementId<3> neighbor_5_id{2, std::array{s_10, s_00, s_00}};

  const ElementId<3> neighbor_6_id{3, std::array{s_10, s_10, s_00}};
  const ElementId<3> neighbor_7_id{3, std::array{s_20, s_22, s_00}};
  const ElementId<3> neighbor_8_id{3, std::array{s_20, s_23, s_00}};
  const ElementId<3> neighbor_12_id{3, std::array{s_10, s_11, s_00}};

  DirectionMap<3, Neighbors<3>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{neighbor_8_id, neighbor_7_id},
                   b3_orientation});
  child_1_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{child_2_id}, aligned});
  child_1_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{neighbor_1_id, neighbor_2_id},
                   b4_orientation});
  child_1_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{child_3_id}, aligned});
  Element<3> child_1{child_1_id, std::move(child_1_neighbors)};
  Mesh<3> child_1_mesh{std::array{3_st, 4_st, 3_st}, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  std::array child_1_flags{amr::Flag::Join, amr::Flag::Join,
                           amr::Flag::IncreaseResolution};

  DirectionMap<3, Neighbors<3>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{neighbor_4_id}, b1_orientation});
  child_2_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{neighbor_3_id}, b4_orientation});
  child_2_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{child_3_id}, aligned});
  Element<3> child_2{child_2_id, std::move(child_2_neighbors)};
  Mesh<3> child_2_mesh{std::array{4_st, 3_st, 4_st}, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  std::array child_2_flags{amr::Flag::Join, amr::Flag::Join,
                           amr::Flag::DecreaseResolution};

  DirectionMap<3, Neighbors<3>> child_3_neighbors{};
  child_3_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{neighbor_6_id}, b3_orientation});
  child_3_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{neighbor_4_id}, b1_orientation});
  child_3_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{child_2_id, child_1_id}, aligned});
  child_3_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{neighbor_5_id}, b2_orientation});
  Element<3> child_3{child_3_id, std::move(child_3_neighbors)};
  Mesh<3> child_3_mesh{std::array{4_st, 4_st, 3_st}, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  std::array child_3_flags{amr::Flag::DecreaseResolution, amr::Flag::Join,
                           amr::Flag::IncreaseResolution};

  std::unordered_map<ElementId<3>, std::array<amr::Flag, 3>>
      child_1_neighbor_flags{
          {neighbor_1_id,
           {{amr::Flag::DoNothing, amr::Flag::Join, amr::Flag::DoNothing}}},
          {neighbor_2_id,
           {{amr::Flag::DoNothing, amr::Flag::Join, amr::Flag::DoNothing}}},
          {child_2_id, child_2_flags},
          {child_3_id, child_3_flags},
          {neighbor_7_id,
           {{amr::Flag::Join, amr::Flag::Join, amr::Flag::DoNothing}}},
          {neighbor_8_id,
           {{amr::Flag::Join, amr::Flag::Join, amr::Flag::DoNothing}}}};
  std::unordered_map<ElementId<3>, std::array<amr::Flag, 3>>
      child_2_neighbor_flags{
          {neighbor_3_id,
           {{amr::Flag::DoNothing, amr::Flag::DoNothing,
             amr::Flag::DoNothing}}},
          {neighbor_4_id,
           {{amr::Flag::DoNothing, amr::Flag::Split, amr::Flag::DoNothing}}},
          {child_2_id, child_2_flags},
          {child_1_id, child_1_flags}};
  std::unordered_map<ElementId<3>, std::array<amr::Flag, 3>>
      child_3_neighbor_flags{
          {child_1_id, child_1_flags},
          {child_2_id, child_2_flags},
          {neighbor_4_id,
           {{amr::Flag::DoNothing, amr::Flag::Split, amr::Flag::DoNothing}}},
          {neighbor_5_id,
           {{amr::Flag::DoNothing, amr::Flag::DoNothing,
             amr::Flag::DoNothing}}},
          {neighbor_6_id,
           {{amr::Flag::DoNothing, amr::Flag::DoNothing,
             amr::Flag::DoNothing}}}};

  using TaggedTupleType =
      tuples::TaggedTuple<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          Parallel::Tags::ArrayIndexImpl<ElementId<3>>,
                          Parallel::Tags::GlobalCacheImpl<Metavariables>,
                          domain::Tags::Element<3>, domain::Tags::Mesh<3>,
                          amr::Tags::Flags<3>, amr::Tags::NeighborFlags<3>>;
  std::unordered_map<ElementId<3>, TaggedTupleType> children_items;
  children_items.emplace(
      child_1_id,
      TaggedTupleType{Metavariables{}, child_1_id, nullptr, std::move(child_1),
                      std::move(child_1_mesh), std::move(child_1_flags),
                      std::move(child_1_neighbor_flags)});
  children_items.emplace(
      child_2_id,
      TaggedTupleType{Metavariables{}, child_2_id, nullptr, std::move(child_2),
                      std::move(child_2_mesh), std::move(child_2_flags),
                      std::move(child_2_neighbor_flags)});
  children_items.emplace(
      child_3_id,
      TaggedTupleType{Metavariables{}, child_3_id, nullptr, std::move(child_3),
                      std::move(child_3_mesh), std::move(child_3_flags),
                      std::move(child_3_neighbor_flags)});

  DirectionMap<3, Neighbors<3>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{neighbor_12_id, neighbor_6_id},
                   b3_orientation});
  expected_parent_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{neighbor_10_id, neighbor_11_id},
                   b1_orientation});
  expected_parent_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{neighbor_9_id, neighbor_3_id},
                   b4_orientation});
  expected_parent_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{neighbor_5_id}, b2_orientation});
  Element<3> expected_parent{parent_id, std::move(expected_parent_neighbors)};

  const Mesh<3> expected_parent_mesh{4, Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto};
  const std::array expected_parent_flags{
      amr::Flag::Undefined, amr::Flag::Undefined, amr::Flag::Undefined};
  const std::unordered_map<ElementId<3>, std::array<amr::Flag, 3>>
      expected_parent_neighbor_flags{};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<my_component>(&runner, parent_id);
  ActionTesting::simple_action<my_component, amr::Actions::InitializeParent>(
      make_not_null(&runner), parent_id, children_items);
  CHECK(ActionTesting::get_databox_tag<my_component, domain::Tags::Element<3>>(
            runner, parent_id) == expected_parent);
  CHECK(ActionTesting::get_databox_tag<my_component, domain::Tags::Mesh<3>>(
            runner, parent_id) == expected_parent_mesh);
  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<3>>(
            runner, parent_id) == expected_parent_flags);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborFlags<3>>(
          runner, parent_id) == expected_parent_neighbor_flags);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.InitializeParent",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
