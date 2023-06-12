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
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::EventsAndTriggers>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 typename Metavariables::simple_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::RunEventsAndTriggers>>>;
};

template <typename SimpleTags = tmpl::list<>>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using simple_tags = SimpleTags;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::Completion>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
};

void run_events_and_triggers(const EventsAndTriggers& events_and_triggers,
                             const bool expected) {
  using my_component = Component<Metavariables<>>;
  ActionTesting::MockRuntimeSystem<Metavariables<>> runner{
      {serialize_and_deserialize(events_and_triggers)}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);

  CHECK(ActionTesting::get_terminate<my_component>(runner, 0) == expected);
}

void check_trigger(const bool expected, const std::string& trigger_string) {
  // Test factory
  auto trigger =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables<>>(
          trigger_string);

  EventsAndTriggers::Storage events_and_triggers_input;
  events_and_triggers_input.push_back(
      {std::move(trigger), make_vector<std::unique_ptr<Event>>(
                               std::make_unique<Events::Completion>())});
  const EventsAndTriggers events_and_triggers(
      std::move(events_and_triggers_input));

  run_events_and_triggers(events_and_triggers, expected);
}

void test_completion() {
  const auto completion =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables<>>(
          "Completion");
  CHECK(not completion->needs_evolved_variables());
}

void test_basic_triggers() {
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

void test_factory() {
  const auto events_and_triggers =
      TestHelpers::test_creation<EventsAndTriggers, Metavariables<>>(
          "- Trigger:\n"
          "    Not: Always\n"
          "  Events:\n"
          "    - Completion\n"
          "- Trigger:\n"
          "    Or:\n"
          "      - Not: Always\n"
          "      - Always\n"
          "  Events:\n"
          "    - Completion\n"
          "    - Completion\n"
          "- Trigger:\n"
          "    Not: Always\n"
          "  Events:\n"
          "    - Completion\n");

  run_events_and_triggers(events_and_triggers, true);
}

void test_slab_limits() {
  EventsAndTriggers::Storage events_and_triggers_input;
  events_and_triggers_input.push_back(
      {std::make_unique<Triggers::Always>(),
       make_vector<std::unique_ptr<Event>>(
           std::make_unique<Events::Completion>())});

  const Slab slab(0.0, 1.0);
  const auto start = slab.start();
  const auto center = start + slab.duration() / 2;

  using metavars = Metavariables<tmpl::list<Tags::TimeStepId>>;
  using my_component = Component<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {EventsAndTriggers(std::move(events_and_triggers_input))}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& box =
      ActionTesting::get_databox<my_component>(make_not_null(&runner), 0);

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, center);
      },
      make_not_null(&box));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  CHECK(not ActionTesting::get_terminate<my_component>(runner, 0));

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, start, 1, slab.duration(), start.value());
      },
      make_not_null(&box));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  CHECK(not ActionTesting::get_terminate<my_component>(runner, 0));

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, start);
      },
      make_not_null(&box));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_terminate<my_component>(runner, 0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndTriggers", "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables<>>();

  test_completion();
  test_basic_triggers();
  test_factory();
  test_slab_limits();
}
