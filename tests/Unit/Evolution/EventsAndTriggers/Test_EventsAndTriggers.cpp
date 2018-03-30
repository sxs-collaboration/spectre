// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file checks the Completion event and the basic logical
// triggers (Always, And, Not, and Or).

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/EventsAndTriggers/Completion.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Evolution/EventsAndTriggers/LogicalTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct DefaultClasses {
  template <typename T>
  using type = tmpl::list<>;
};

using events_and_triggers_tag =
    Tags::EventsAndTriggers<DefaultClasses, DefaultClasses>;

struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int,
                                      tmpl::list<events_and_triggers_tag>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
};

using EventsAndTriggersType = EventsAndTriggers<DefaultClasses, DefaultClasses>;

void run_events_and_triggers(const EventsAndTriggersType& events_and_triggers,
                             const bool expected) {
  // Test pup
  Parallel::register_derived_classes_with_charm<Event<DefaultClasses>>();
  Parallel::register_derived_classes_with_charm<Trigger<DefaultClasses>>();
  ActionTesting::ActionRunner<Metavariables> runner{{
    serialize_and_deserialize(events_and_triggers)}};

  const auto box = db::create<db::AddSimpleTags<>>();

  runner.apply<component, Actions::RunEventsAndTriggers>(box, 0);

  CHECK(runner.algorithms<component>()[0].get_terminate() == expected);
}

void check_trigger(const bool expected, const std::string& trigger_string) {
  // Test factory
  std::unique_ptr<Trigger<DefaultClasses>> trigger =
      test_factory_creation<Trigger<DefaultClasses>>(trigger_string);

  EventsAndTriggersType::Storage events_and_triggers_map;
  events_and_triggers_map.emplace(
      std::move(trigger),
      make_vector<std::unique_ptr<Event<DefaultClasses>>>(
          std::make_unique<Events::Completion<DefaultClasses>>()));
  const EventsAndTriggersType events_and_triggers(
      std::move(events_and_triggers_map));

  run_events_and_triggers(events_and_triggers, expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers", "[Unit][Evolution]") {
  test_factory_creation<Event<DefaultClasses>>("  Completion");

  check_trigger(true,
                "  Always");
  check_trigger(false,
                "  Not: Always");
  check_trigger(true,
                "  Not:\n"
                "    Not: Always");

  check_trigger(true,
                "  And:\n"
                "    - Always\n"
                "    - Always");
  check_trigger(false,
                "  And:\n"
                "    - Always\n"
                "    - Not: Always");
  check_trigger(false,
                "  And:\n"
                "    - Not: Always\n"
                "    - Always");
  check_trigger(false,
                "  And:\n"
                "    - Not: Always\n"
                "    - Not: Always");
  check_trigger(false,
                "  And:\n"
                "    - Always\n"
                "    - Always\n"
                "    - Not: Always");

  check_trigger(true,
                "  Or:\n"
                "    - Always\n"
                "    - Always");
  check_trigger(true,
                "  Or:\n"
                "    - Always\n"
                "    - Not: Always");
  check_trigger(true,
                "  Or:\n"
                "    - Not: Always\n"
                "    - Always");
  check_trigger(false,
                "  Or:\n"
                "    - Not: Always\n"
                "    - Not: Always");
  check_trigger(true,
                "  Or:\n"
                "    - Not: Always\n"
                "    - Not: Always\n"
                "    - Always");
}

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers.creation",
                  "[Unit][Evolution]") {
  const auto events_and_triggers = test_creation<EventsAndTriggersType>(
      "  ? Not: Always\n"
      "  : - Completion\n"
      "  ? Or:\n"
      "    - Not: Always\n"
      "    - Always\n"
      "  : - Completion\n"
      "    - Completion\n"
      "  ? Not: Always\n"
      "  : - Completion\n");

  run_events_and_triggers(events_and_triggers, true);
}
