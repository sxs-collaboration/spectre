// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <utility>

#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct TestEvent : public Event {
 public:
  explicit TestEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestEvent);  // NOLINT
#pragma GCC diagnostic pop

  using compute_tags_for_observation_box = tmpl::list<>;

  TestEvent() = default;

  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& observation_value) const {
    CHECK(observation_value.name == "IterationId(Label)");
    CHECK(observation_value.value == 10.0);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};

PUP::able::PUP_ID TestEvent::my_PUP_ID = 0;  // NOLINT

struct Label {};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<Convergence::Tags::IterationId<Label>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<elliptic::Actions::RunEventsAndTriggers<
                                 Convergence::Tags::IterationId<Label>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<TestEvent>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.RunEventsAndTriggers", "[Unit][Elliptic]") {
  register_factory_classes_with_charm<Metavariables>();

  EventsAndTriggers::Storage events_and_triggers_input;
  events_and_triggers_input.push_back(
      {std::make_unique<Triggers::Always>(),
       make_vector<std::unique_ptr<Event>>(std::make_unique<TestEvent>())});

  using my_component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {EventsAndTriggers(std::move(events_and_triggers_input))}};
  ActionTesting::emplace_component_and_initialize<my_component>(&runner, 0,
                                                                {10_st});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
}
