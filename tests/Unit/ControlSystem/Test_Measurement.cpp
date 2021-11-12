// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <type_traits>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Event.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using MeasureEvent = control_system::Event<
    tmpl::list<control_system::TestHelpers::ExampleControlSystem>>;

template <typename Metavariables>
struct ElementComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockControlSystemComponent {
  using component_being_mocked =
      ControlComponent<Metavariables,
                       control_system::TestHelpers::ExampleControlSystem>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using initialization_tags = tmpl::list<
      control_system::TestHelpers::ExampleControlSystem::MeasurementQueue,
      control_system::TestHelpers::MeasurementResultTime,
      control_system::TestHelpers::MeasurementResultTag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockControlSystemComponent<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<MeasureEvent>>>;
  };

  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

// This test tests control_system::Event and
// control_system::RunCallbacks, and also does additional testing of
// the protocol examples.
SPECTRE_TEST_CASE("Unit.ControlSystem.Measurement", "[ControlSystem][Unit]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using element_component = ElementComponent<Metavariables>;
  using control_system_component = MockControlSystemComponent<Metavariables>;
  const element_component* const element_component_p = nullptr;

  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_singleton_component<control_system_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0},
      control_system::TestHelpers::ExampleControlSystem::MeasurementQueue::
          type{},
      0.0, 0.0);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  auto& cache = ActionTesting::cache<element_component>(runner, 0);
  // Making our own DataBox is easier than using the one in the mock
  // element.
  const auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::Time, evolution::Tags::PreviousTriggerTime,
                        control_system::TestHelpers::SomeTagOnElement>>(
      Metavariables{}, 1.234, std::optional<double>{}, 5.678);

  const MeasureEvent event{};
  CHECK(event.needs_evolved_variables());
  event.run(make_observation_box<db::AddComputeTags<>>(box), cache, 0,
            element_component_p);

  ActionTesting::invoke_queued_simple_action<control_system_component>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::is_simple_action_queue_empty<control_system_component>(
      runner, 0));

  CHECK(ActionTesting::get_databox_tag<
            control_system_component,
            control_system::TestHelpers::MeasurementResultTime>(
            runner, 0) == 1.234);
  CHECK(ActionTesting::get_databox_tag<
            control_system_component,
            control_system::TestHelpers::MeasurementResultTag>(runner, 0) ==
        5.678);
}
