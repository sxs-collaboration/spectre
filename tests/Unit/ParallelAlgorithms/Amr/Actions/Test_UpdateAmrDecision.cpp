// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
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
  using simple_tags = tmpl::list<domain::Tags::Element<volume_dim>,
                                 amr::Tags::Flags<volume_dim>,
                                 amr::Tags::NeighborFlags<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<Component<Metavariables>>;
};

void check(const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
           const ElementId<1>& id,
           const std::array<amr::Flag, 1>& expected_flags,
           const std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>&
               expected_neighbor_flags,
           const size_t expected_queued_actions) {
  using my_component = Component<Metavariables>;
  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<1>>(
            runner, id) == expected_flags);
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborFlags<1>>(
          runner, id) == expected_neighbor_flags);
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, id) == expected_queued_actions);
}

// When AMR is run, the simple action EvaluateAmrCriteria is run on each
// Element.  EvaluateAmrCriteria evaluates the criteria which determine the
// amr::Tags::Flags of the Element, and then calls the simple action
// UpdateAmrDecision on each neighboring Element of the Element sending the
// Flags. UpdateAmrDecision checks to see if an Elements Flags need to change
// based on the received NeighborFlags (e.g. if and element wants to join, but
// its sibling does not the element must change its decision to do nothing).  If
// the element's Flags are changed, then it calls UpdateAmrDecision on its
// neighbors, and the process continues until no Element wants to change its
// decision.   This test manually runs this process on three elements assuming
// each Element has already evaluted its own Flags with the ones passed to
// emplace_component_and_initialize.
void test() {
  using my_component = Component<Metavariables>;

  const ElementId<1> self_id(0, {{{2, 1}}});
  const ElementId<1> sibling_id(0, {{{2, 0}}});
  const ElementId<1> cousin_id(1, {{{2, 2}}});

  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      initial_neighbor_flags{};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  const Element<1> self(self_id,
                        {{{Direction<1>::lower_xi(), {{sibling_id}, {}}},
                          {Direction<1>::upper_xi(), {{cousin_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {self, std::array{amr::Flag::Join}, initial_neighbor_flags});

  const Element<1> sibling(sibling_id,
                           {{{Direction<1>::upper_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, sibling_id,
      {sibling, std::array{amr::Flag::Join}, initial_neighbor_flags});

  const Element<1> cousin(cousin_id,
                          {{{Direction<1>::lower_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, cousin_id,
      {cousin, std::array{amr::Flag::Split}, initial_neighbor_flags});

  check(runner, self_id, {{amr::Flag::Join}}, {}, 0);
  check(runner, sibling_id, {{amr::Flag::Join}}, {}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}}, {}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), self_id, sibling_id, std::array{amr::Flag::Join});
  // sibling told self it wants to join, no further action triggered
  check(runner, self_id, {{amr::Flag::Join}},
        {{sibling_id, {{amr::Flag::Join}}}}, 0);
  check(runner, sibling_id, {{amr::Flag::Join}}, {}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}}, {}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), sibling_id, self_id, std::array{amr::Flag::Join});
  // self told sibling it wants to join, no further action triggered
  check(runner, self_id, {{amr::Flag::Join}},
        {{sibling_id, {{amr::Flag::Join}}}}, 0);
  check(runner, sibling_id, {{amr::Flag::Join}},
        {{self_id, {{amr::Flag::Join}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}}, {}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), cousin_id, self_id, std::array{amr::Flag::Join});
  // self told cousin it wants to join, no further action triggered
  check(runner, self_id, {{amr::Flag::Join}},
        {{sibling_id, {{amr::Flag::Join}}}}, 0);
  check(runner, sibling_id, {{amr::Flag::Join}},
        {{self_id, {{amr::Flag::Join}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}},
        {{self_id, {{amr::Flag::Join}}}}, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), self_id, cousin_id, std::array{amr::Flag::Split});
  // cousin told self it wants to split; in order to maintain 2:1 refinement,
  // self changes its decision to DoNothing and triggers actions to notify its
  // neighbors
  check(runner, self_id, {{amr::Flag::DoNothing}},
        {{sibling_id, {{amr::Flag::Join}}}, {cousin_id, {{amr::Flag::Split}}}},
        0);
  check(runner, sibling_id, {{amr::Flag::Join}},
        {{self_id, {{amr::Flag::Join}}}}, 1);
  check(runner, cousin_id, {{amr::Flag::Split}},
        {{self_id, {{amr::Flag::Join}}}}, 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), sibling_id);
  // self told sibling it wants to DoNothing, so sibling changes its decision to
  // DoNothing and triggers actions to notify its neighbors
  check(runner, self_id, {{amr::Flag::DoNothing}},
        {{sibling_id, {{amr::Flag::Join}}}, {cousin_id, {{amr::Flag::Split}}}},
        1);
  check(runner, sibling_id, {{amr::Flag::DoNothing}},
        {{self_id, {{amr::Flag::DoNothing}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}},
        {{self_id, {{amr::Flag::Join}}}}, 1);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), cousin_id);
  // self told cousin it wants to DoNothing, no further action triggered
  check(runner, self_id, {{amr::Flag::DoNothing}},
        {{sibling_id, {{amr::Flag::Join}}}, {cousin_id, {{amr::Flag::Split}}}},
        1);
  check(runner, sibling_id, {{amr::Flag::DoNothing}},
        {{self_id, {{amr::Flag::DoNothing}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}},
        {{self_id, {{amr::Flag::DoNothing}}}}, 0);

  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), self_id);
  // sibling told self it wants to DoNothing, no further action triggered
  check(runner, self_id, {{amr::Flag::DoNothing}},
        {{sibling_id, {{amr::Flag::DoNothing}}},
         {cousin_id, {{amr::Flag::Split}}}},
        0);
  check(runner, sibling_id, {{amr::Flag::DoNothing}},
        {{self_id, {{amr::Flag::DoNothing}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}},
        {{self_id, {{amr::Flag::DoNothing}}}}, 0);
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

  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      initial_neighbor_flags{};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  const Element<1> self(self_id,
                        {{{Direction<1>::lower_xi(), {{sibling_id}, {}}},
                          {Direction<1>::upper_xi(), {{cousin_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {self, std::array{amr::Flag::Split},
       std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>{
           {cousin_id, {{amr::Flag::Split}}}}});

  const Element<1> sibling(sibling_id,
                           {{{Direction<1>::upper_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, sibling_id,
      {sibling, std::array{amr::Flag::Undefined}, initial_neighbor_flags});

  const Element<1> cousin(cousin_id,
                          {{{Direction<1>::lower_xi(), {{self_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, cousin_id,
      {cousin, std::array{amr::Flag::Split}, initial_neighbor_flags});

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), sibling_id, self_id,
      std::array{amr::Flag::Split});
  // self told sibling it wants to split, but sibling hasn't evaluated its own
  // flags, store the received flags and exit
  check(runner, sibling_id, {{amr::Flag::Undefined}},
        {{self_id, {{amr::Flag::Split}}}}, 0);
  check(runner, self_id, {{amr::Flag::Split}},
        {{cousin_id, {{amr::Flag::Split}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}}, initial_neighbor_flags, 0);

  ActionTesting::simple_action<my_component, amr::Actions::UpdateAmrDecision>(
      make_not_null(&runner), self_id, cousin_id,
      std::array{amr::Flag::DoNothing});
  // cousin first chose to do nothing, then chose to split, but the messages
  // were received in the reverse order. Therefore we ignore the lower priority
  // message of doing nothing.
  check(runner, sibling_id, {{amr::Flag::Undefined}},
        {{self_id, {{amr::Flag::Split}}}}, 0);
  check(runner, self_id, {{amr::Flag::Split}},
        {{cousin_id, {{amr::Flag::Split}}}}, 0);
  check(runner, cousin_id, {{amr::Flag::Split}}, initial_neighbor_flags, 0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.UpdateAmrDecision",
                  "[Unit][ParallelAlgorithms]") {
  test();
  test_race_conditions();
}
