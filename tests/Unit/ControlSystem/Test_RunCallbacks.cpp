// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <type_traits>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct LabelA {};
using Measurement = control_system::TestHelpers::Measurement<LabelA, true>;
struct System : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "A"; }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = Measurement;
  using simple_tags = tmpl::list<>;
  using control_error = control_system::TestHelpers::ControlError<1>;
  static constexpr size_t deriv_order = 2;

  struct SubmeasurementQueueTag {
    using type = double;
  };

  struct MeasurementQueue : db::SimpleTag {
    using type = LinkedMessageQueue<double, tmpl::list<SubmeasurementQueueTag>>;
  };

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags =
        tmpl::list<control_system::TestHelpers::MeasurementResultTag>;

    template <typename Submeasurement, typename Metavariables>
    static void apply(Submeasurement /*meta*/, const double measurement_result,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const LinkedMessageId<double>& measurement_id) {
      // Process the submeasurement results and send whatever is
      // necessary to the control system component.  Usually calls
      // some simple action.
      auto& control_system_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, System>>(cache);
      Parallel::simple_action<::Actions::UpdateMessageQueue<
          SubmeasurementQueueTag, MeasurementQueue,
          control_system::TestHelpers::SomeControlSystemUpdater>>(
          control_system_proxy, measurement_id, measurement_result);
    }
  };
};
static_assert(
    tt::assert_conforms_to_v<System, control_system::protocols::ControlSystem>);
// avoid unused variable warning
static_assert(System::deriv_order == 2);

using MeasureEvent = tmpl::front<
    typename Measurement::submeasurements>::template event<tmpl::list<System>>;

template <typename Metavariables>
struct ElementComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockControlSystemComponent {
  using component_being_mocked = ControlComponent<Metavariables, System>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<control_system::Tags::Verbosity>;
  using simple_tags_from_options =
      tmpl::list<System::MeasurementQueue,
                 control_system::TestHelpers::MeasurementResultTime,
                 control_system::TestHelpers::MeasurementResultTag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockControlSystemComponent<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<MeasureEvent>>>;
  };
};
}  // namespace

// This tests control_system::RunCallbacks and also does additional testing of
// the protocol examples.
SPECTRE_TEST_CASE("Unit.ControlSystem.RunCallbacks", "[ControlSystem][Unit]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using element_component = ElementComponent<Metavariables>;
  using control_system_component = MockControlSystemComponent<Metavariables>;
  const element_component* const element_component_p = nullptr;

  MockRuntimeSystem runner{{::Verbosity::Silent}};
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_singleton_component<control_system_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, System::MeasurementQueue::type{}, 0.0,
      0.0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<element_component>(runner, 0);
  // Making our own DataBox is easier than using the one in the mock
  // element.
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::Time, evolution::Tags::PreviousTriggerTime,
                        control_system::TestHelpers::SomeTagOnElement>>(
      Metavariables{}, 1.234, std::optional<double>{}, 5.678);

  const MeasureEvent event{};
  auto obs_box = make_observation_box<db::AddComputeTags<
      control_system::TestHelpers::SomeOtherTagOnElementCompute>>(
      make_not_null(&box));
  event.run(make_not_null(&obs_box), cache, 0, element_component_p, {});

  ActionTesting::invoke_queued_simple_action<control_system_component>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::is_simple_action_queue_empty<control_system_component>(
      runner, 0));

  CHECK(ActionTesting::get_databox_tag<
            control_system_component,
            control_system::TestHelpers::MeasurementResultTime>(runner, 0) ==
        1.234);
  CHECK(ActionTesting::get_databox_tag<
            control_system_component,
            control_system::TestHelpers::MeasurementResultTag>(runner, 0) ==
        5.678);
}
