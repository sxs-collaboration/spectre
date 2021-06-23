// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/EventsAndDenseTriggers/DenseTriggers/TestTrigger.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace TriggerLabels {
struct A {};
struct B {};
}  // namespace TriggerLabels

using TriggerA = TestHelpers::DenseTriggers::BoxTrigger<TriggerLabels::A>;
using TriggerB = TestHelpers::DenseTriggers::BoxTrigger<TriggerLabels::B>;

template <typename Label>
class TestEvent : public Event {
 public:
  explicit TestEvent(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestEvent);  // NOLINT

  static std::string name() noexcept {
    return "TestEvent<" + Options::name<Label>() + ">";
  }

  using options = tmpl::list<>;
  static constexpr Options::String help = "help";

  TestEvent() = default;

  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/) const noexcept {
    event_ran = true;
  }

  struct IsReady : db::SimpleTag {
    using type = bool;
  };

  using is_ready_argument_tags = tmpl::list<IsReady>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(const bool ready,
                Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const noexcept {
    return ready;
  }

  bool needs_evolved_variables() const noexcept override {
    return Label::needs_evolved_variables;
  }

  static bool event_ran;
};

template <typename Label>
bool TestEvent<Label>::event_ran = false;

template <typename Label>
PUP::able::PUP_ID TestEvent<Label>::my_PUP_ID = 0;  // NOLINT

namespace EventLabels {
struct A {
  static constexpr bool needs_evolved_variables = false;
};
struct B {
  static constexpr bool needs_evolved_variables = false;
};
struct C {
  static constexpr bool needs_evolved_variables = true;
};
}  // namespace EventLabels

using EventA = TestEvent<EventLabels::A>;
using EventB = TestEvent<EventLabels::B>;
using EventC = TestEvent<EventLabels::C>;

struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DenseTrigger, tmpl::list<TriggerA, TriggerB>>,
                  tmpl::pair<Event, tmpl::list<EventA, EventB, EventC>>>;
  };
};

void do_test(const bool time_runs_forward) noexcept {
  const double time_sign = time_runs_forward ? 1.0 : -1.0;

  Parallel::GlobalCache<Metavariables> cache{};
  const int array_index = 0;
  const int* component = nullptr;

  auto events_and_dense_triggers =
      TestHelpers::test_creation<evolution::EventsAndDenseTriggers,
                                 Metavariables>(
          "BoxTrigger<A>:\n"
          "  - TestEvent<A>\n"
          "BoxTrigger<B>:\n"
          "  - TestEvent<B>\n"
          "  - TestEvent<C>\n");

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::TimeStepId,
      Tags::Time, TriggerA::IsReady, TriggerA::IsTriggered, TriggerA::NextCheck,
      TriggerB::IsReady, TriggerB::IsTriggered, TriggerB::NextCheck,
      EventA::IsReady, EventB::IsReady, EventC::IsReady>>(
      Metavariables{}, TimeStepId(time_runs_forward, 0, Slab(0.0, 1.0).start()),
      -1.0 * time_sign, false, false, 2.0 * time_sign, false, false,
      3.0 * time_sign, false, false, false);

  const auto set_tag = [&box](auto tag_v, const auto value) noexcept {
    using Tag = decltype(tag_v);
    db::mutate<Tag>(
        make_not_null(&box),
        [&value](const gsl::not_null<typename Tag::type*> var) noexcept {
          *var = value;
        });
  };

  const auto check_events = [](const bool expected_a, const bool expected_b,
                               const bool expected_c) noexcept {
    CHECK(EventA::event_ran == expected_a);
    CHECK(EventB::event_ran == expected_b);
    CHECK(EventC::event_ran == expected_c);
    EventA::event_ran = false;
    EventB::event_ran = false;
    EventC::event_ran = false;
  };

  using TriggeringState = evolution::EventsAndDenseTriggers::TriggeringState;
  CHECK(events_and_dense_triggers.next_trigger(box) == -1.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::NotReady);
  set_tag(TriggerA::IsReady{}, true);
  CHECK(events_and_dense_triggers.next_trigger(box) == -1.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::NotReady);
  set_tag(TriggerB::IsReady{}, true);
  CHECK(events_and_dense_triggers.next_trigger(box) == -1.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::Ready);
  events_and_dense_triggers.run_events(box, cache, array_index, component);
  check_events(false, false, false);

  CHECK(events_and_dense_triggers.next_trigger(box) == 2.0 * time_sign);
  set_tag(Tags::Time{}, 2.0 * time_sign);
  set_tag(TriggerA::IsReady{}, false);
  set_tag(TriggerB::IsReady{}, false);
  CHECK(events_and_dense_triggers.next_trigger(box) == 2.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::NotReady);
  set_tag(TriggerA::IsReady{}, true);
  set_tag(TriggerA::IsTriggered{}, true);
  set_tag(TriggerA::NextCheck{}, 3.0 * time_sign);
  CHECK(events_and_dense_triggers.next_trigger(box) == 2.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::NotReady);
  set_tag(EventA::IsReady{}, true);
  CHECK(events_and_dense_triggers.next_trigger(box) == 2.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::Ready);
  events_and_dense_triggers.run_events(box, cache, array_index, component);
  check_events(true, false, false);

  set_tag(TriggerA::NextCheck{}, 4.0 * time_sign);
  set_tag(TriggerB::IsReady{}, true);
  set_tag(TriggerB::IsTriggered{}, true);
  set_tag(TriggerB::NextCheck{}, 4.0 * time_sign);
  CHECK(events_and_dense_triggers.next_trigger(box) == 3.0 * time_sign);
  set_tag(Tags::Time{}, 3.0 * time_sign);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::NotReady);
  set_tag(EventB::IsReady{}, true);
  CHECK(events_and_dense_triggers.is_ready(
            box, cache, array_index, component) == TriggeringState::NotReady);
  set_tag(EventC::IsReady{}, true);
  CHECK(
      events_and_dense_triggers.is_ready(box, cache, array_index, component) ==
      TriggeringState::NeedsEvolvedVariables);

  const auto finish_checks =
      [&array_index, &box, &cache, &check_events, &component,
       &time_sign](evolution::EventsAndDenseTriggers eadt) noexcept {
        CHECK(eadt.next_trigger(box) == 3.0 * time_sign);
        CHECK(eadt.is_ready(box, cache, array_index, component) ==
              TriggeringState::NeedsEvolvedVariables);
        // In a real executable we might need to wait for data to do
        // dense output at this point.
        CHECK(eadt.next_trigger(box) == 3.0 * time_sign);
        CHECK(eadt.is_ready(box, cache, array_index, component) ==
              TriggeringState::NeedsEvolvedVariables);
        eadt.run_events(box, cache, array_index, component);
        check_events(true, true, true);
      };

  finish_checks(serialize_and_deserialize(events_and_dense_triggers));
  finish_checks(std::move(events_and_dense_triggers));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers",
                  "[Unit][Evolution]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  do_test(true);
  do_test(false);
}
