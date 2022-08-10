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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Amr/Actions/EvaluateRefinementCriteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// when called on the specified refinement level, this criteria
// always will choose to join
auto create_always_join(const size_t refinement_level) {
  return std::make_unique<amr::Criteria::Random>(1.0, refinement_level);
}

// when called on any refinement level, this criteria always will choose to do
// nothing
auto create_always_do_nothing() {
  return std::make_unique<amr::Criteria::Random>(
      0.0, ElementId<3>::max_refinement_level);
}

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<amr::Criteria::Tags::Criteria>;
  using simple_tags = tmpl::list<domain::Tags::Element<volume_dim>,
                                 amr::domain::Tags::Flags<volume_dim>,
                                 amr::domain::Tags::NeighborFlags<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;

  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion, tmpl::list<amr::Criteria::Random>>>;
  };
};

// When AMR is run, the simple action EvaluateAmrCriteria is run on each
// Element.  EvaluateAmrCriteria evaluates the criteria which set its own
// amr::domain::Tags::Flags and then calls the simple action UpdateAmrDecision
// on each neighboring Element of the Element sending the Flags.
// UpdateAmrDecision checks to see if an Elements Flags need to change based on
// the received NeighborFlags (e.g. if and element wants to join, but its
// sibling does not the element must change its decision to do nothing).  If the
// element's Flags are changed, then it calls UpdateAmrDecision on its
// neighbors, and the process continues until no Element wants to change its
// decision.   This test manually runs this process on three elements until
// EvaluateAmrCriteria has been called on each Element.  Note in a asynchronus
// parallel environment, it is possible for an Element to execute
// UpdateAmrDecision (triggered by a neighboring Element) prior to executing
// EvaluateAmrCriteria
void evaluate_criteria(std::vector<std::unique_ptr<amr::Criterion>> criteria,
                       const std::array<amr::domain::Flag, 1> expected_flags) {
  using my_component = Component<Metavariables>;

  const ElementId<1> self_id(0, {{{1, 1}}});
  const ElementId<1> lo_id(0, {{{1, 0}}});
  const ElementId<1> up_id(1, {{{1, 0}}});

  std::unordered_map<ElementId<1>, std::array<amr::domain::Flag, 1>>
      initial_neighbor_flags;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{std::move(criteria)}};

  const Element<1> self(self_id, {{{Direction<1>::lower_xi(), {{lo_id}, {}}},
                                   {Direction<1>::upper_xi(), {{up_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {self, std::array{amr::domain::Flag::Undefined}, initial_neighbor_flags});

  const auto emplace_neighbor = [&self_id, &runner, &initial_neighbor_flags](
                                    const ElementId<1>& id,
                                    const Direction<1>& direction) {
    const Element<1> element(id, {{direction, {{self_id}, {}}}});
    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, id,
        {element, std::array{amr::domain::Flag::Undefined},
         initial_neighbor_flags});
  };

  emplace_neighbor(up_id, Direction<1>::lower_xi());
  emplace_neighbor(lo_id, Direction<1>::upper_xi());

  runner.set_phase(Parallel::Phase::Testing);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::Flags<1>>(
              runner, id) == std::array{amr::domain::Flag::Undefined});
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::NeighborFlags<1>>(
              runner, id) == initial_neighbor_flags);
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }

  // self runs EvaluateAmrCriteria, queueing UpdateAmrDecision on lo and hi
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), self_id);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::Flags<1>>(
              runner, id) == (id == self_id
                                  ? expected_flags
                                  : std::array{amr::domain::Flag::Undefined}));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::NeighborFlags<1>>(
              runner, id) == initial_neighbor_flags);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == self_id ? 0 : 1));
  }

  // lo runs EvaluateAmrCriteria, queuing UpdateAmrDecision on self
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), lo_id);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::Flags<1>>(
              runner, id) == (id == up_id
                                  ? std::array{amr::domain::Flag::Undefined}
                                  : expected_flags));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::NeighborFlags<1>>(
              runner, id) == initial_neighbor_flags);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == 1);
  }

  // up runs UpdateAmrDecision, which queues nothing
  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), up_id);
  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::Flags<1>>(
              runner, id) == (id == up_id
                                  ? std::array{amr::domain::Flag::Undefined}
                                  : expected_flags));
    CHECK(
        ActionTesting::get_databox_tag<my_component,
                                       amr::domain::Tags::NeighborFlags<1>>(
            runner, id) ==
        (id == up_id
             ? std::unordered_map<ElementId<1>, std::array<amr::domain::Flag,
                                                           1>>{{self_id,
                                                                expected_flags}}
             : initial_neighbor_flags));
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == up_id ? 0 : 1));
  }

  // up runs EvaluateAmrCriteria, queueing UpdateAmrDecision on self
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), up_id);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::domain::Tags::Flags<1>>(
              runner, id) == expected_flags);
    CHECK(
        ActionTesting::get_databox_tag<my_component,
                                       amr::domain::Tags::NeighborFlags<1>>(
            runner, id) ==
        (id == up_id
             ? std::unordered_map<ElementId<1>, std::array<amr::domain::Flag,
                                                           1>>{{self_id,
                                                                expected_flags}}
             : initial_neighbor_flags));
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == self_id ? 2 : (id == lo_id ? 1 : 0)));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.EvaluateRefinementCriteria",
                  "[Unit][ParallelAlgorithms]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  // Run the test 3 times, twice with a single criterion that give known
  // decisions, and then once with two criteria, one of which always produces
  // flags of a higher priority than the other
  criteria.emplace_back(create_always_join(1));
  evaluate_criteria(std::move(criteria), std::array{amr::domain::Flag::Join});
  criteria.clear();
  criteria.emplace_back(create_always_do_nothing());
  criteria.emplace_back(create_always_join(1));
  evaluate_criteria(std::move(criteria),
                    std::array{amr::domain::Flag::DoNothing});
  criteria.clear();
  criteria.emplace_back(create_always_join(1));
  criteria.emplace_back(create_always_do_nothing());
  evaluate_criteria(std::move(criteria),
                    std::array{amr::domain::Flag::DoNothing});
}
