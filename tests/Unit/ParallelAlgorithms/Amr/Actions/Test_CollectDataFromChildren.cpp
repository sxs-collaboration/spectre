// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <deque>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Amr/Helpers.hpp"
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
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
ElementId<2> parent_id{0, std::array{SegmentId{0, 0}, SegmentId{0, 0}}};
ElementId<2> child_1_id{0, std::array{SegmentId{1, 0}, SegmentId{1, 0}}};
ElementId<2> child_2_id{0, std::array{SegmentId{1, 1}, SegmentId{1, 0}}};
ElementId<2> child_3_id{0, std::array{SegmentId{1, 0}, SegmentId{1, 1}}};
ElementId<2> child_4_id{0, std::array{SegmentId{1, 1}, SegmentId{1, 1}}};
ElementId<2> neighbor_1_id{1, std::array{SegmentId{1, 1}, SegmentId{0, 0}}};
ElementId<2> neighbor_2_id{2, std::array{SegmentId{1, 0}, SegmentId{0, 0}}};
ElementId<2> neighbor_3_id{2, std::array{SegmentId{1, 1}, SegmentId{1, 1}}};

auto child_1_mesh() {
  return Mesh<2>{std::array{3_st, 3_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}
auto child_2_mesh() {
  return Mesh<2>{std::array{3_st, 4_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}
auto child_3_mesh() {
  return Mesh<2>{std::array{4_st, 3_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}
auto child_4_mesh() {
  return Mesh<2>{std::array{4_st, 4_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

auto child_flags() { return std::array{amr::Flag::Join, amr::Flag::Join}; }
auto neighbor_1_flags() {
  return std::array{amr::Flag::Join, amr::Flag::DoNothing};
}

Element<2> child_1() {
  static Element<2> result{
      child_1_id,
      DirectionMap<2, Neighbors<2>>{
          {Direction<2>::lower_xi(),
           Neighbors<2>{std::unordered_set{neighbor_1_id},
                        OrientationMap<2>{}}},
          {Direction<2>::upper_xi(),
           Neighbors<2>{std::unordered_set{child_2_id}, OrientationMap<2>{}}},
          {Direction<2>::lower_eta(),
           Neighbors<2>{std::unordered_set{child_3_id}, OrientationMap<2>{}}},
          {Direction<2>::upper_eta(),
           Neighbors<2>{std::unordered_set{child_3_id}, OrientationMap<2>{}}}}};
  return result;
}

Element<2> child_2() {
  static Element<2> result{
      child_2_id,
      DirectionMap<2, Neighbors<2>>{
          {Direction<2>::lower_xi(),
           Neighbors<2>{std::unordered_set{child_1_id}, OrientationMap<2>{}}},
          {Direction<2>::upper_xi(),
           Neighbors<2>{
               std::unordered_set{neighbor_2_id},
               OrientationMap<2>{std::array{Direction<2>::lower_eta(),
                                            Direction<2>::upper_xi()}}}},
          {Direction<2>::lower_eta(),
           Neighbors<2>{std::unordered_set{child_4_id}, OrientationMap<2>{}}},
          {Direction<2>::upper_eta(),
           Neighbors<2>{std::unordered_set{child_4_id}, OrientationMap<2>{}}}}};
  return result;
}

Element<2> child_3() {
  static Element<2> result{
      child_3_id,
      DirectionMap<2, Neighbors<2>>{
          {Direction<2>::lower_xi(),
           Neighbors<2>{std::unordered_set{neighbor_1_id},
                        OrientationMap<2>{}}},
          {Direction<2>::upper_xi(),
           Neighbors<2>{std::unordered_set{child_4_id}, OrientationMap<2>{}}},
          {Direction<2>::lower_eta(),
           Neighbors<2>{std::unordered_set{child_1_id}, OrientationMap<2>{}}},
          {Direction<2>::upper_eta(),
           Neighbors<2>{std::unordered_set{child_1_id}, OrientationMap<2>{}}}}};
  return result;
}

Element<2> child_4() {
  static Element<2> result{
      child_4_id,
      DirectionMap<2, Neighbors<2>>{
          {Direction<2>::lower_xi(),
           Neighbors<2>{std::unordered_set{child_3_id}, OrientationMap<2>{}}},
          {Direction<2>::upper_xi(),
           Neighbors<2>{
               std::unordered_set{neighbor_3_id},
               OrientationMap<2>{std::array{Direction<2>::lower_eta(),
                                            Direction<2>::upper_xi()}}}},
          {Direction<2>::lower_eta(),
           Neighbors<2>{std::unordered_set{child_2_id}, OrientationMap<2>{}}},
          {Direction<2>::upper_eta(),
           Neighbors<2>{std::unordered_set{child_2_id}, OrientationMap<2>{}}}}};
  return result;
}

auto child_1_neighbor_flags() {
  return std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>{
      {neighbor_1_id, neighbor_1_flags()},
      {child_2_id, child_flags()},
      {child_3_id, child_flags()}};
}

auto child_2_neighbor_flags() {
  return std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>{
      {child_1_id, child_flags()},
      {child_4_id, child_flags()},
      {neighbor_2_id, std::array{amr::Flag::DoNothing, amr::Flag::Split}}};
}

auto child_3_neighbor_flags() {
  return std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>{
      {neighbor_1_id, neighbor_1_flags()},
      {child_1_id, child_flags()},
      {child_4_id, child_flags()}};
}
auto child_4_neighbor_flags() {
  return std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>{
      {child_2_id, child_flags()},
      {child_3_id, child_flags()},
      {neighbor_3_id, std::array{amr::Flag::DoNothing, amr::Flag::Join}}};
}

struct MockInitializeParent {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename... Tags>
  static void apply(
      DataBox& /*box*/, const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& mock_parent_id,
      const std::unordered_map<ElementId<Metavariables::volume_dim>,
                               tuples::TaggedTuple<Tags...>>& children_items) {
    CHECK(mock_parent_id == parent_id);
    const auto& child_1_items = children_items.at(child_1_id);
    CHECK(get<domain::Tags::Element<2>>(child_1_items) == child_1());
    CHECK(get<domain::Tags::Mesh<2>>(child_1_items) == child_1_mesh());
    CHECK(get<amr::Tags::Flags<2>>(child_1_items) == child_flags());
    CHECK(get<amr::Tags::NeighborFlags<2>>(child_1_items) ==
          child_1_neighbor_flags());

    const auto& child_2_items = children_items.at(child_2_id);
    CHECK(get<domain::Tags::Element<2>>(child_2_items) == child_2());
    CHECK(get<domain::Tags::Mesh<2>>(child_2_items) == child_2_mesh());
    CHECK(get<amr::Tags::Flags<2>>(child_2_items) == child_flags());
    CHECK(get<amr::Tags::NeighborFlags<2>>(child_2_items) ==
          child_2_neighbor_flags());

    const auto& child_3_items = children_items.at(child_3_id);
    CHECK(get<domain::Tags::Element<2>>(child_3_items) == child_3());
    CHECK(get<domain::Tags::Mesh<2>>(child_3_items) == child_3_mesh());
    CHECK(get<amr::Tags::Flags<2>>(child_3_items) == child_flags());
    CHECK(get<amr::Tags::NeighborFlags<2>>(child_3_items) ==
          child_3_neighbor_flags());

    const auto& child_4_items = children_items.at(child_4_id);
    CHECK(get<domain::Tags::Element<2>>(child_4_items) == child_4());
    CHECK(get<domain::Tags::Mesh<2>>(child_4_items) == child_4_mesh());
    CHECK(get<amr::Tags::Flags<2>>(child_4_items) == child_flags());
    CHECK(get<amr::Tags::NeighborFlags<2>>(child_4_items) ==
          child_4_neighbor_flags());
  }
};

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
  using replace_these_simple_actions =
      tmpl::list<amr::Actions::InitializeParent>;
  using with_these_simple_actions = tmpl::list<MockInitializeParent>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<Component<Metavariables>>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

void test() {
  using my_component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, child_1_id,
      {child_1(), child_1_mesh(), child_flags(), child_1_neighbor_flags()});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, child_2_id,
      {child_2(), child_2_mesh(), child_flags(), child_2_neighbor_flags()});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, child_3_id,
      {child_3(), child_3_mesh(), child_flags(), child_3_neighbor_flags()});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, child_4_id,
      {child_4(), child_4_mesh(), child_flags(), child_4_neighbor_flags()});
  ActionTesting::emplace_component<my_component>(&runner, parent_id);

  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, child_4_id, parent_id}) {
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }

  ActionTesting::simple_action<my_component,
                               amr::Actions::CollectDataFromChildren>(
      make_not_null(&runner), child_1_id, parent_id,
      std::deque{child_2_id, child_3_id});
  for (const auto& id :
       std::vector{child_1_id, child_3_id, child_4_id, parent_id}) {
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, child_2_id) == 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), child_2_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_4_id, parent_id}) {
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, child_3_id) == 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), child_3_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, parent_id}) {
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, child_4_id) == 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), child_4_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, child_4_id}) {
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, parent_id) == 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), parent_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, child_4_id, parent_id}) {
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.CollectDataFromChildren",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
