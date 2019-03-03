// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file checks the Completion event and the basic logical
// triggers (Always, And, Not, and Or).

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Completion.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Evolution/EventsAndTriggers/LogicalTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Tags.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

// IWYU pragma: no_forward_declare db::DataBox

namespace {
using events_and_triggers_tag =
    OptionTags::EventsAndTriggers<tmpl::list<>, tmpl::list<>>;

struct Metavariables;
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<events_and_triggers_tag>;
  using action_list = tmpl::list<Actions::RunEventsAndTriggers>;
  using initial_databox = db::DataBox<tmpl::list<>>;
};

struct Metavariables {
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};

using EventsAndTriggersType = EventsAndTriggers<tmpl::list<>, tmpl::list<>>;

void run_events_and_triggers(const EventsAndTriggersType& events_and_triggers,
                             const bool expected) {
  // Test pup
  Parallel::register_derived_classes_with_charm<Event<tmpl::list<>>>();
  Parallel::register_derived_classes_with_charm<Trigger<tmpl::list<>>>();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using my_component = component;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          my_component>;
  typename MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0, db::DataBox<tmpl::list<>>{});
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {serialize_and_deserialize(events_and_triggers)},
      std::move(dist_objects)};

  runner.next_action<component>(0);

  CHECK(runner.algorithms<component>()[0].get_terminate() == expected);
}

void check_trigger(const bool expected, const std::string& trigger_string) {
  // Test factory
  std::unique_ptr<Trigger<tmpl::list<>>> trigger =
      test_factory_creation<Trigger<tmpl::list<>>>(trigger_string);

  EventsAndTriggersType::Storage events_and_triggers_map;
  events_and_triggers_map.emplace(
      std::move(trigger),
      make_vector<std::unique_ptr<Event<tmpl::list<>>>>(
          std::make_unique<Events::Completion<tmpl::list<>>>()));
  const EventsAndTriggersType events_and_triggers(
      std::move(events_and_triggers_map));

  run_events_and_triggers(events_and_triggers, expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers", "[Unit][Evolution]") {
  test_factory_creation<Event<tmpl::list<>>>("  Completion");

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
