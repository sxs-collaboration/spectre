// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file checks the Completion event and the basic logical
// triggers (Always, And, Not, and Or).

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::EventsAndTriggers>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<Actions::RunEventsAndTriggers>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::Completion>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
  enum class Phase { Initialization, Testing, Exit };
};

void run_events_and_triggers(const EventsAndTriggers& events_and_triggers,
                             const bool expected) {
  // Test pup
  Parallel::register_factory_classes_with_charm<Metavariables>();

  using my_component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {serialize_and_deserialize(events_and_triggers)}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);

  CHECK(ActionTesting::get_terminate<my_component>(runner, 0) == expected);
}

void check_trigger(const bool expected, const std::string& trigger_string) {
  // Test factory
  auto trigger =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
          trigger_string);

  EventsAndTriggers::Storage events_and_triggers_map;
  events_and_triggers_map.emplace(std::move(trigger),
                                  make_vector<std::unique_ptr<Event>>(
                                      std::make_unique<Events::Completion>()));
  const EventsAndTriggers events_and_triggers(
      std::move(events_and_triggers_map));

  run_events_and_triggers(events_and_triggers, expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers", "[Unit][Evolution]") {
  {
    const auto completion =
        TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
            "Completion");
    CHECK(not completion->needs_evolved_variables());
  }

  check_trigger(true, "Always");
  check_trigger(false, "Not: Always");
  check_trigger(true,
                "Not:\n"
                "  Not: Always");

  check_trigger(true,
                "And:\n"
                "  - Always\n"
                "  - Always");
  check_trigger(false,
                "And:\n"
                "  - Always\n"
                "  - Not: Always");
  check_trigger(false,
                "And:\n"
                "  - Not: Always\n"
                "  - Always");
  check_trigger(false,
                "And:\n"
                "  - Not: Always\n"
                "  - Not: Always");
  check_trigger(false,
                "And:\n"
                "  - Always\n"
                "  - Always\n"
                "  - Not: Always");

  check_trigger(true,
                "Or:\n"
                "  - Always\n"
                "  - Always");
  check_trigger(true,
                "Or:\n"
                "  - Always\n"
                "  - Not: Always");
  check_trigger(true,
                "Or:\n"
                "  - Not: Always\n"
                "  - Always");
  check_trigger(false,
                "Or:\n"
                "  - Not: Always\n"
                "  - Not: Always");
  check_trigger(true,
                "Or:\n"
                "  - Not: Always\n"
                "  - Not: Always\n"
                "  - Always");
}

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers.creation",
                  "[Unit][Evolution]") {
  const auto events_and_triggers =
      TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
          "? Not: Always\n"
          ": - Completion\n"
          "? Or:\n"
          "  - Not: Always\n"
          "  - Always\n"
          ": - Completion\n"
          "  - Completion\n"
          "? Not: Always\n"
          ": - Completion\n");

  run_events_and_triggers(events_and_triggers, true);
}
