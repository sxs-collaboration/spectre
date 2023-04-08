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
#include "ParallelAlgorithms/Amr/Actions/EvaluateRefinementCriteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
  using simple_tags =
      tmpl::list<domain::Tags::Element<volume_dim>,
                 domain::Tags::Mesh<volume_dim>, amr::Tags::Flags<volume_dim>,
                 amr::Tags::NeighborFlags<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <size_t VolumeDim>
struct Metavariables {
  static constexpr size_t volume_dim = VolumeDim;

  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        amr::Criterion, tmpl::list<amr::Criteria::Random,
                                   amr::Criteria::DriveToTarget<volume_dim>>>>;
  };
};

// When AMR is run, the simple action EvaluateAmrCriteria is run on each
// Element.  EvaluateAmrCriteria evaluates the criteria which set its own
// amr::Tags::Flags and then calls the simple action UpdateAmrDecision
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
                       const std::array<amr::Flag, 1> expected_flags) {
  using my_component = Component<Metavariables<1>>;

  const ElementId<1> self_id(0, {{{1, 1}}});
  const ElementId<1> lo_id(0, {{{1, 0}}});
  const ElementId<1> up_id(1, {{{1, 0}}});
  const Mesh<1> mesh{2_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  std::unordered_map<ElementId<1>, std::array<amr::Flag, 1>>
      initial_neighbor_flags;

  ActionTesting::MockRuntimeSystem<Metavariables<1>> runner{
      {std::move(criteria)}};

  const Element<1> self(self_id, {{{Direction<1>::lower_xi(), {{lo_id}, {}}},
                                   {Direction<1>::upper_xi(), {{up_id}, {}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {self, mesh, std::array{amr::Flag::Undefined}, initial_neighbor_flags});

  const auto emplace_neighbor =
      [&self_id, &mesh, &runner, &initial_neighbor_flags](
          const ElementId<1>& id, const Direction<1>& direction) {
        const Element<1> element(id, {{direction, {{self_id}, {}}}});
        ActionTesting::emplace_component_and_initialize<my_component>(
            &runner, id,
            {element, mesh, std::array{amr::Flag::Undefined},
             initial_neighbor_flags});
      };

  emplace_neighbor(up_id, Direction<1>::lower_xi());
  emplace_neighbor(lo_id, Direction<1>::upper_xi());

  runner.set_phase(Parallel::Phase::Testing);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<1>>(
              runner, id) == std::array{amr::Flag::Undefined});
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborFlags<1>>(
              runner, id) == initial_neighbor_flags);
    CHECK(
        ActionTesting::is_simple_action_queue_empty<my_component>(runner, id));
  }

  // self runs EvaluateAmrCriteria, queueing UpdateAmrDecision on lo and hi
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), self_id);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<1>>(
              runner, id) ==
          (id == self_id ? expected_flags : std::array{amr::Flag::Undefined}));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborFlags<1>>(
              runner, id) == initial_neighbor_flags);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == self_id ? 0 : 1));
  }

  // lo runs EvaluateAmrCriteria, queuing UpdateAmrDecision on self
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), lo_id);

  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<1>>(
              runner, id) ==
          (id == up_id ? std::array{amr::Flag::Undefined} : expected_flags));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborFlags<1>>(
              runner, id) == initial_neighbor_flags);
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == 1);
  }

  // up runs UpdateAmrDecision, which queues nothing
  ActionTesting::invoke_queued_simple_action<my_component>(
      make_not_null(&runner), up_id);
  for (const auto& id : {self_id, lo_id, up_id}) {
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<1>>(
              runner, id) ==
          (id == up_id ? std::array{amr::Flag::Undefined} : expected_flags));
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborFlags<1>>(runner,
                                                                      id) ==
          (id == up_id
               ? std::unordered_map<ElementId<1>,
                                    std::array<amr::Flag, 1>>{{self_id,
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
    CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<1>>(
              runner, id) == expected_flags);
    CHECK(ActionTesting::get_databox_tag<my_component,
                                         amr::Tags::NeighborFlags<1>>(runner,
                                                                      id) ==
          (id == up_id
               ? std::unordered_map<ElementId<1>,
                                    std::array<amr::Flag, 1>>{{self_id,
                                                               expected_flags}}
               : initial_neighbor_flags));
    CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
              runner, id) == (id == self_id ? 2 : (id == lo_id ? 1 : 0)));
  }
}

void check_split_while_join_is_avoided() {
  using my_component = Component<Metavariables<2>>;

  // The part of action we are testing does not depend upon information
  // from neighbors, so we just use a single Element setup on refinement
  // levels (0, 1)
  const ElementId<2> self_id(0, {{{0, 0}, {1, 1}}});
  const Mesh<2> mesh{2_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  std::unordered_map<ElementId<2>, std::array<amr::Flag, 2>>
      initial_neighbor_flags{};

  // the refinement criteria wants to drive self to levels (1, 0) so
  // it will return flags (Split, Join).
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  criteria.emplace_back(std::make_unique<amr::Criteria::DriveToTarget<2>>(
      std::array{2_st, 2_st}, std::array{1_st, 0_st},
      std::array{amr::Flag::DoNothing, amr::Flag::DoNothing}));

  Parallel::GlobalCache<Metavariables<2>> empty_cache{};
  const auto databox = db::create<tmpl::list<::domain::Tags::Mesh<2>>>(mesh);
  ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<::domain::Tags::Mesh<2>>>>
      box{databox};
  auto flags_from_criterion =
      criteria.front()->evaluate(box, empty_cache, self_id);
  CHECK(flags_from_criterion == std::array{amr::Flag::Split, amr::Flag::Join});

  // But we do not allow an Element to simultaneously split and join so the
  // action should change the flags to (DoNothing, Split)
  ActionTesting::MockRuntimeSystem<Metavariables<2>> runner{
      {std::move(criteria)}};

  const Element<2> self(self_id, {});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {self, mesh, std::array{amr::Flag::Undefined, amr::Flag::Undefined},
       initial_neighbor_flags});

  runner.set_phase(Parallel::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<2>>(
            runner, self_id) ==
        std::array{amr::Flag::Undefined, amr::Flag::Undefined});
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborFlags<2>>(
          runner, self_id) == initial_neighbor_flags);
  CHECK(ActionTesting::is_simple_action_queue_empty<my_component>(runner,
                                                                  self_id));

  // self runs EvaluateAmrCriteria
  ActionTesting::simple_action<my_component,
                               amr::Actions::EvaluateRefinementCriteria>(
      make_not_null(&runner), self_id);

  CHECK(ActionTesting::get_databox_tag<my_component, amr::Tags::Flags<2>>(
            runner, self_id) ==
        std::array{amr::Flag::Split, amr::Flag::DoNothing});
  CHECK(
      ActionTesting::get_databox_tag<my_component, amr::Tags::NeighborFlags<2>>(
          runner, self_id) == initial_neighbor_flags);
  CHECK(ActionTesting::number_of_queued_simple_actions<my_component>(
            runner, self_id) == 0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.EvaluateRefinementCriteria",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables<2>>();
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  // Run the test 3 times, twice with a single criterion that give known
  // decisions, and then once with two criteria, one of which always produces
  // flags of a higher priority than the other
  criteria.emplace_back(create_always_join(1));
  evaluate_criteria(std::move(criteria), std::array{amr::Flag::Join});
  criteria.clear();
  criteria.emplace_back(create_always_do_nothing());
  criteria.emplace_back(create_always_join(1));
  evaluate_criteria(std::move(criteria), std::array{amr::Flag::DoNothing});
  criteria.clear();
  criteria.emplace_back(create_always_join(1));
  criteria.emplace_back(create_always_do_nothing());
  evaluate_criteria(std::move(criteria), std::array{amr::Flag::DoNothing});
  check_split_while_join_is_avoided();
}
