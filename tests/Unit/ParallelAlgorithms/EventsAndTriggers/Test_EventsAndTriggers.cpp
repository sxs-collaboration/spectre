// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file checks EventsAndTriggers and the basic logical
// triggers (Always, And, Not, and Or).

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
struct Data : db::SimpleTag {
  using type = int;
};

struct Observation : db::SimpleTag {
  using type = int;
};

struct ObservationCompute : Observation, db::ComputeTag {
  using base = Observation;
  using argument_tags = tmpl::list<Data>;
  static void function(const gsl::not_null<int*> observation, const int data) {
    *observation = data + 1;
  }
};

struct RunCount : db::SimpleTag {
  using type = int;
};
}  // namespace Tags

struct TestEvent : public Event {
 public:
  explicit TestEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestEvent);  // NOLINT
#pragma GCC diagnostic pop

  using compute_tags_for_observation_box = tmpl::list<Tags::ObservationCompute>;
  using options = tmpl::list<>;
  static constexpr Options::String help = "";

  TestEvent() = default;

  using return_tags = tmpl::list<Tags::RunCount>;
  using argument_tags = tmpl::list<Tags::Data, Tags::Observation>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(const gsl::not_null<int*> run_count, const int data,
                  const int observation,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& observation_value) const {
    CHECK(data == 2);
    CHECK(observation == 3);
    CHECK(observation_value.name == "Name");
    CHECK(observation_value.value == 1234.5);
    ++*run_count;
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

struct Component {};

struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<TestEvent>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
};

void run_events_and_triggers(const EventsAndTriggers& events_and_triggers,
                             const int expected) {
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::Data, Tags::RunCount>>(Metavariables{}, 2, 0);
  Event::ObservationValue observation_value{"Name", 1234.5};

  Parallel::GlobalCache<Metavariables> cache{};
  Component* const component_ptr = nullptr;

  events_and_triggers.run_events(make_not_null(&box), cache, 0, component_ptr,
                                 observation_value);
  CHECK(db::get<Tags::RunCount>(box) == expected);

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  serialize_and_deserialize(events_and_triggers)
      .run_events(make_not_null(&box), cache, 0, component_ptr,
                  observation_value);
  CHECK(db::get<Tags::RunCount>(box) == expected);
}

void check_trigger(const int expected, const std::string& trigger_string) {
  auto trigger =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
          trigger_string);

  EventsAndTriggers::Storage events_and_triggers_input;
  events_and_triggers_input.push_back(
      {std::move(trigger),
       make_vector<std::unique_ptr<Event>>(std::make_unique<TestEvent>())});
  const EventsAndTriggers events_and_triggers(
      std::move(events_and_triggers_input));

  run_events_and_triggers(events_and_triggers, expected);
}

void test_basic_triggers() {
  check_trigger(1, "Always");
  check_trigger(0, "Not: Always");
  check_trigger(1,
                "Not:\n"
                "  Not: Always");

  check_trigger(1,
                "And:\n"
                "  - Always\n"
                "  - Always");
  check_trigger(0,
                "And:\n"
                "  - Always\n"
                "  - Not: Always");
  check_trigger(0,
                "And:\n"
                "  - Not: Always\n"
                "  - Always");
  check_trigger(0,
                "And:\n"
                "  - Not: Always\n"
                "  - Not: Always");
  check_trigger(0,
                "And:\n"
                "  - Always\n"
                "  - Always\n"
                "  - Not: Always");

  check_trigger(1,
                "Or:\n"
                "  - Always\n"
                "  - Always");
  check_trigger(1,
                "Or:\n"
                "  - Always\n"
                "  - Not: Always");
  check_trigger(1,
                "Or:\n"
                "  - Not: Always\n"
                "  - Always");
  check_trigger(0,
                "Or:\n"
                "  - Not: Always\n"
                "  - Not: Always");
  check_trigger(1,
                "Or:\n"
                "  - Not: Always\n"
                "  - Not: Always\n"
                "  - Always");
}

void test_factory() {
  const auto events_and_triggers =
      TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
          "- Trigger:\n"
          "    Not: Always\n"
          "  Events:\n"
          "    - TestEvent\n"
          "- Trigger:\n"
          "    Or:\n"
          "      - Not: Always\n"
          "      - Always\n"
          "  Events:\n"
          "    - TestEvent\n"
          "    - TestEvent\n"
          "- Trigger:\n"
          "    Not: Always\n"
          "  Events:\n"
          "    - TestEvent\n");

  run_events_and_triggers(events_and_triggers, 2);
}

void test_custom_check_trigger() {
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::Data, Tags::RunCount>>(Metavariables{}, 2, 0);
  Event::ObservationValue observation_value{"Name", 1234.5};

  Parallel::GlobalCache<Metavariables> cache{};
  Component* const component_ptr = nullptr;

  const EventsAndTriggers always_eat = []() {
    EventsAndTriggers::Storage events_and_triggers_input;
    events_and_triggers_input.push_back(
        {std::make_unique<Triggers::Always>(),
         make_vector<std::unique_ptr<Event>>(std::make_unique<TestEvent>())});
    return EventsAndTriggers(std::move(events_and_triggers_input));
  }();
  const EventsAndTriggers never_eat = []() {
    EventsAndTriggers::Storage events_and_triggers_input;
    events_and_triggers_input.push_back(
        {std::make_unique<Triggers::Not>(std::make_unique<Triggers::Always>()),
         make_vector<std::unique_ptr<Event>>(std::make_unique<TestEvent>())});
    return EventsAndTriggers(std::move(events_and_triggers_input));
  }();

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  always_eat.run_events(make_not_null(&box), cache, 0, component_ptr,
                        observation_value);
  CHECK(db::get<Tags::RunCount>(box) == 1);

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  never_eat.run_events(make_not_null(&box), cache, 0, component_ptr,
                       observation_value);
  CHECK(db::get<Tags::RunCount>(box) == 0);

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  always_eat.run_events(make_not_null(&box), cache, 0, component_ptr,
                        observation_value,
                        [](const Trigger& /*trigger*/) { return false; });
  CHECK(db::get<Tags::RunCount>(box) == 0);

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  never_eat.run_events(make_not_null(&box), cache, 0, component_ptr,
                       observation_value,
                       [](const Trigger& /*trigger*/) { return false; });
  CHECK(db::get<Tags::RunCount>(box) == 0);

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  always_eat.run_events(make_not_null(&box), cache, 0, component_ptr,
                        observation_value,
                        [](const Trigger& /*trigger*/) { return true; });
  CHECK(db::get<Tags::RunCount>(box) == 1);

  db::mutate<Tags::RunCount>(
      [](const gsl::not_null<int*> run_count) { *run_count = 0; },
      make_not_null(&box));
  never_eat.run_events(make_not_null(&box), cache, 0, component_ptr,
                       observation_value,
                       [](const Trigger& /*trigger*/) { return true; });
  CHECK(db::get<Tags::RunCount>(box) == 1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.EventsAndTriggers",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();

  test_basic_triggers();
  test_factory();
  test_custom_check_trigger();
}
