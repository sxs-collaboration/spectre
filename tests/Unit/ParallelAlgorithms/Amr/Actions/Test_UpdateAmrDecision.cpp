// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/UpdateAmrDecision.hpp"
#include "Utilities/StdHelpers.hpp"
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
      tmpl::list<domain::Tags::Mesh<volume_dim>,
                 domain::Tags::Element<volume_dim>, amr::Tags::Info<volume_dim>,
                 amr::Tags::NeighborInfo<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<Component<Metavariables>>;
};

void check(const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
           const ElementId<1>& id, const amr::Info<1>& expected_info,
           const std::unordered_map<ElementId<1>, amr::Info<1>>&
               expected_neighbor_info,
           const size_t expected_queued_actions) {
  using my_component = Component<Metavariables>;
  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Info<1>>(
            runner, id) == expected_info);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborInfo<1>>(
          runner, id) == expected_neighbor_info);
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, id) == expected_queued_actions);
}

// When AMR is run, the simple action EvaluateAmrCriteria is run on each
// Element.  EvaluateAmrCriteria evaluates the criteria which determine the
// amr::Tags::Info of the Element, and then calls the simple action
// UpdateAmrDecision on each neighboring Element of the Element sending the
// Info. UpdateAmrDecision checks to see if an Elements Info need to change
// based on the received NeighborInfo (e.g. if and element wants to join, but
// its sibling does not the element must change its decision to do nothing).  If
// the element's Info are changed, then it calls UpdateAmrDecision on its
// neighbors, and the process continues until no Element wants to change its
// decision.   This test manually runs this process on three elements assuming
// each Element has already evaluted its own Info with the ones passed to
// emplace_component_and_initialize.
void test() {
  using my_component = Component<Metavariables>;

  const ElementId<1> self_id(0, {{{2, 1}}});
  const ElementId<1> sibling_id(0, {{{2, 0}}});
  const ElementId<1> cousin_id(1, {{{2, 2}}});
  const Mesh<1> mesh{2_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  std::unordered_map<ElementId<1>, amr::Info<1>> initial_neighbor_info{};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  const Element<1> self(self_id,
                        {{{Direction<1>::lower_xi(), {{sibling_id}, {}}},
                          {Direction<1>::upper_xi(), {{cousin_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {mesh, self, amr::Info<1>{std::array{amr::Flag::Join}, mesh},
       initial_neighbor_info});

  const Element<1> sibling(sibling_id,
                           {{{Direction<1>::upper_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, sibling_id,
      {mesh, sibling, amr::Info<1>{std::array{amr::Flag::Join}, mesh},
       initial_neighbor_info});

  const Element<1> cousin(cousin_id,
                          {{{Direction<1>::lower_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, cousin_id,
      {mesh, cousin, amr::Info<1>{std::array{amr::Flag::Split}, mesh},
       initial_neighbor_info});

  check(runner, self_id, {{{amr::Flag::Join}}, mesh}, {}, 0);
  check(runner, sibling_id, {{{amr::Flag::Join}}, mesh}, {}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh}, {}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), self_id, sibling_id,
      amr::Info<1>{std::array{amr::Flag::Join}, mesh});
  // sibling told self it wants to join, no further action triggered
  check(runner, self_id, {{{amr::Flag::Join}}, mesh},
        {{sibling_id, {{{amr::Flag::Join}}, mesh}}}, 0);
  check(runner, sibling_id, {{{amr::Flag::Join}}, mesh}, {}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh}, {}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), sibling_id, self_id,
      amr::Info<1>{std::array{amr::Flag::Join}, mesh});
  // self told sibling it wants to join, no further action triggered
  check(runner, self_id, {{{amr::Flag::Join}}, mesh},
        {{sibling_id, {{{amr::Flag::Join}}, mesh}}}, 0);
  check(runner, sibling_id, {{{amr::Flag::Join}}, mesh},
        {{self_id, {{{amr::Flag::Join}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh}, {}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), cousin_id, self_id,
      amr::Info<1>{std::array{amr::Flag::Join}, mesh});
  // self told cousin it wants to join, no further action triggered
  check(runner, self_id, {{{amr::Flag::Join}}, mesh},
        {{sibling_id, {{{amr::Flag::Join}}, mesh}}}, 0);
  check(runner, sibling_id, {{{amr::Flag::Join}}, mesh},
        {{self_id, {{{amr::Flag::Join}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh},
        {{self_id, {{{amr::Flag::Join}}, mesh}}}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), self_id, cousin_id,
      amr::Info<1>{std::array{amr::Flag::Split}, mesh});
  // cousin told self it wants to split; in order to maintain 2:1 refinement,
  // self changes its decision to DoNothing and triggers actions to notify its
  // neighbors
  check(runner, self_id, {{{amr::Flag::DoNothing}}, mesh},
        {{sibling_id, {{{amr::Flag::Join}}, mesh}},
         {cousin_id, {{{amr::Flag::Split}}, mesh}}},
        0);
  check(runner, sibling_id, {{{amr::Flag::Join}}, mesh},
        {{self_id, {{{amr::Flag::Join}}, mesh}}}, 1);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh},
        {{self_id, {{{amr::Flag::Join}}, mesh}}}, 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), sibling_id);
  // self told sibling it wants to DoNothing, so sibling changes its decision to
  // DoNothing and triggers actions to notify its neighbors
  check(runner, self_id, {{{amr::Flag::DoNothing}}, mesh},
        {{sibling_id, {{{amr::Flag::Join}}, mesh}},
         {cousin_id, {{{amr::Flag::Split}}, mesh}}},
        1);
  check(runner, sibling_id, {{{amr::Flag::DoNothing}}, mesh},
        {{self_id, {{{amr::Flag::DoNothing}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh},
        {{self_id, {{{amr::Flag::Join}}, mesh}}}, 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), cousin_id);
  // self told cousin it wants to DoNothing, no further action triggered
  check(runner, self_id, {{{amr::Flag::DoNothing}}, mesh},
        {{sibling_id, {{{amr::Flag::Join}}, mesh}},
         {cousin_id, {{{amr::Flag::Split}}, mesh}}},
        1);
  check(runner, sibling_id, {{{amr::Flag::DoNothing}}, mesh},
        {{self_id, {{{amr::Flag::DoNothing}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh},
        {{self_id, {{{amr::Flag::DoNothing}}, mesh}}}, 0);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), self_id);
  // sibling told self it wants to DoNothing, no further action triggered
  check(runner, self_id, {{{amr::Flag::DoNothing}}, mesh},
        {{sibling_id, {{{amr::Flag::DoNothing}}, mesh}},
         {cousin_id, {{{amr::Flag::Split}}, mesh}}},
        0);
  check(runner, sibling_id, {{{amr::Flag::DoNothing}}, mesh},
        {{self_id, {{{amr::Flag::DoNothing}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh},
        {{self_id, {{{amr::Flag::DoNothing}}, mesh}}}, 0);
}

// This test checks that asynchronus execution of simple actions is handled
// correctly in case UpdateAmrDecision is called on an Element before
// EvaluateAmrCriteria, or if two calls to UpdateAmrDecision are triggered by
// receiving messages from a neighbor in the opposite order in which they were
// sent
void test_race_conditions() {
  using my_component = Component<Metavariables>;

  const ElementId<1> self_id(0, {{{2, 1}}});
  const ElementId<1> sibling_id(0, {{{2, 0}}});
  const ElementId<1> cousin_id(1, {{{2, 2}}});
  const Mesh<1> mesh{2_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  std::unordered_map<ElementId<1>, amr::Info<1>> initial_neighbor_info{};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  const Element<1> self(self_id,
                        {{{Direction<1>::lower_xi(), {{sibling_id}, {}}},
                          {Direction<1>::upper_xi(), {{cousin_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {mesh, self, amr::Info<1>{std::array{amr::Flag::Split}, mesh},
       std::unordered_map<ElementId<1>, amr::Info<1>>{
           {cousin_id, {{{amr::Flag::Split}}, mesh}}}});

  const Element<1> sibling(sibling_id,
                           {{{Direction<1>::upper_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, sibling_id,
      {mesh, sibling, amr::Info<1>{std::array{amr::Flag::Undefined}, Mesh<1>{}},
       initial_neighbor_info});

  const Element<1> cousin(cousin_id,
                          {{{Direction<1>::lower_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, cousin_id,
      {mesh, cousin, amr::Info<1>{std::array{amr::Flag::Split}, mesh},
       initial_neighbor_info});

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), sibling_id, self_id,
      amr::Info<1>{std::array{amr::Flag::Split}, mesh});
  // self told sibling it wants to split, but sibling hasn't evaluated its own
  // flags, store the received flags and exit
  check(runner, sibling_id, {{{amr::Flag::Undefined}}, Mesh<1>{}},
        {{self_id, {{{amr::Flag::Split}}, mesh}}}, 0);
  check(runner, self_id, {{{amr::Flag::Split}}, mesh},
        {{cousin_id, {{{amr::Flag::Split}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh}, initial_neighbor_info,
        0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), self_id, cousin_id,
      amr::Info<1>{std::array{amr::Flag::DoNothing}, mesh});
  // cousin first chose to do nothing, then chose to split, but the messages
  // were received in the reverse order. Therefore we ignore the lower priority
  // message of doing nothing.
  check(runner, sibling_id, {{{amr::Flag::Undefined}}, Mesh<1>{}},
        {{self_id, {{{amr::Flag::Split}}, mesh}}}, 0);
  check(runner, self_id, {{{amr::Flag::Split}}, mesh},
        {{cousin_id, {{{amr::Flag::Split}}, mesh}}}, 0);
  check(runner, cousin_id, {{{amr::Flag::Split}}, mesh}, initial_neighbor_info,
        0);
}

// When AMR is run, the simple action EvaluateAmrCriteria is run on
// each Element.  EvaluateAmrCriteria evaluates the criteria which
// determine the amr::Tags::Info of the Element, and then calls the
// simple action UpdateAmrDecision on each neighboring Element of the
// Element sending the Info. UpdateAmrDecision checks to see if an
// Elements Info need to change based on the received NeighborInfo
// (e.g. if and element wants to join, but its sibling does not the
// element must change its decision to do nothing).  If the element's
// Info are changed, then it calls UpdateAmrDecision on its neighbors,
// and the process continues until no Element wants to change its
// decision.  This test manually runs this process on a set of
// elements assuming each Element has already evaluted its own Info
// with the ones passed to emplace_component_and_initialize.
//
// - We will create 7 elements in 1D labeled by Ids 0-6 from left to right
// - N is the number of grid points of the initial element
// - NewN is number of grid points the element (or its child/parent) will have
//   after refinement
// - Initial F is the amr::Flag corresponding to that chosen by
//   EvaluateRefinementCriteria (which is assumed to have already been done)
// - Final F is the amr::Flag the element should have after the chain of
//   UpdateAmrDecision actions are executed
// - J = Join, DR = DecreaseResolution, DN = DoNothing,
//   IR = IncreaseResolution, S = Split
//
// Initial   Final state  Note
// Id N F    NewN F
//  0 7 DR    6   DR
//  1 4 S     4   S
//  2 6 J     6   DN      2 and 3 cannot join at Lev0 as 1 split into Lev2
//  3 5 J     5   DN
//  4 2 J     4   J       Joined 4 and 5 gets larger Mesh of the two
//  5 4 J     4   J
//  6 4 IR    5   IR
template <typename Generator>
void test_mesh_update(gsl::not_null<Generator*> generator) {
  using my_component = Component<Metavariables>;

  const auto join = std::array{amr::Flag::Join};
  const auto restrict = std::array{amr::Flag::DecreaseResolution};
  const auto stay = std::array{amr::Flag::DoNothing};
  const auto prolong = std::array{amr::Flag::IncreaseResolution};
  const auto split = std::array{amr::Flag::Split};

  // seven elements, first six at refinement level 1, last at level 0
  const std::vector<ElementId<1>> ids{
      {0, {{{1, 0}}}}, {0, {{{1, 1}}}}, {1, {{{1, 0}}}}, {1, {{{1, 1}}}},
      {2, {{{1, 0}}}}, {2, {{{1, 1}}}}, {3, {{{0, 0}}}}};

  std::vector<Element<1>> elements;
  elements.reserve(7_st);
  elements.emplace_back(
      Element<1>(ids[0], {{{Direction<1>::upper_xi(), {{ids[1]}, {}}}}}));
  for (size_t i = 1; i < 6; ++i) {
    elements.emplace_back(
        Element<1>(ids[i], {{{Direction<1>::lower_xi(), {{ids[i - 1]}, {}}},
                             {Direction<1>::upper_xi(), {{ids[i + 1]}, {}}}}}));
  }
  elements.emplace_back(
      Element<1>(ids[6], {{{Direction<1>::lower_xi(), {{ids[5]}, {}}}}}));

  const auto initial_extents =
      std::vector{7_st, 4_st, 6_st, 5_st, 2_st, 4_st, 4_st};

  std::vector<Mesh<1>> initial_meshes;
  initial_meshes.reserve(7_st);
  for (size_t i = 0; i < 7; ++i) {
    initial_meshes.emplace_back(initial_extents[i], Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto);
  }

  const auto initial_flags =
      std::vector{restrict, split, join, join, join, join, prolong};

  std::vector<amr::Info<1>> initial_infos;
  initial_infos.reserve(7_st);
  // first element is restricted, so needs mesh with extent 6
  initial_infos.emplace_back(amr::Info<1>{initial_flags[0], initial_meshes[2]});
  for (size_t i = 1; i < 6; ++i) {
    initial_infos.emplace_back(
        amr::Info<1>{initial_flags[i], initial_meshes[i]});
  }
  // last element is prolonged, so needs new_mesh with extent 5
  initial_infos.emplace_back(amr::Info<1>{initial_flags[6], initial_meshes[3]});

  std::unordered_map<ElementId<1>, amr::Info<1>> initial_neighbor_info;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  // initialize components assuming EvaluateAmrCriteria has been executed
  for (size_t i = 0; i < 7; ++i) {
    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, ids[i],
        {initial_meshes[i], elements[i], initial_infos[i],
         initial_neighbor_info});
  }

  // manually queue the actions that EvaluateAmrCriteria would have called
  // this sends the amr::Info of an element to its neighbors
  for (size_t i = 1; i < 7; ++i) {
    ActionTesting::queue_simple_action<my_component,
                                       amr::Actions::UpdateAmrDecision>(
        make_not_null(&runner), ids[i], ids[i - 1], initial_infos[i - 1]);
    ActionTesting::queue_simple_action<my_component,
                                       amr::Actions::UpdateAmrDecision>(
        make_not_null(&runner), ids[i - 1], ids[i], initial_infos[i]);
  }

  // Now execute a random queued action on a random component until no component
  // has queued actions
  auto array_indices_with_queued_simple_actions =
      ActionTesting::array_indices_with_queued_simple_actions<
          typename Metavariables::component_list>(make_not_null(&runner));

  while (ActionTesting::number_of_elements_with_queued_simple_actions<
             typename Metavariables::component_list>(
             array_indices_with_queued_simple_actions) > 0) {
    ActionTesting::invoke_random_queued_simple_action<
        typename Metavariables::component_list>(
        make_not_null(&runner), generator,
        array_indices_with_queued_simple_actions);
    array_indices_with_queued_simple_actions =
        ActionTesting::array_indices_with_queued_simple_actions<
            typename Metavariables::component_list>(make_not_null(&runner));
  }

  // check the final state of the info and neighbor info on each element
  const auto expected_new_extents =
      std::vector{6_st, 4_st, 6_st, 5_st, 4_st, 4_st, 5_st};
  const auto expected_flags =
      std::vector{restrict, split, stay, stay, join, join, prolong};
  std::vector<amr::Info<1>> expected_infos;
  expected_infos.reserve(7_st);
  for (size_t i = 0; i < 7; ++i) {
    expected_infos.emplace_back(
        amr::Info<1>{expected_flags[i],
                     Mesh<1>(expected_new_extents[i], Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto)});
  }

  std::vector<std::unordered_map<ElementId<1>, amr::Info<1>>>
      expected_neighbor_infos{7_st};
  for (size_t i = 1; i < 7; ++i) {
    expected_neighbor_infos[i - 1].emplace(ids[i], expected_infos[i]);
    expected_neighbor_infos[i].emplace(ids[i - 1], expected_infos[i - 1]);
  }

  for (size_t i = 0; i < 7; ++i) {
    check(runner, ids[i], expected_infos[i], expected_neighbor_infos[i], 0);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.UpdateAmrDecision",
                  "[Unit][ParallelAlgorithms]") {
  MAKE_GENERATOR(generator);
  test();
  test_race_conditions();
  for (size_t i = 0; i < 1000; ++i) {
    test_mesh_update(make_not_null(&generator));
  }
}
