// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/OnSubsteps.hpp"
#include "Utilities/Gsl.hpp"
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
  using options = tmpl::list<>;
  static constexpr Options::String help = "";

  TestEvent() = default;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& observation_value) const {
    CHECK(observation_value.name == "Time");
    last_value.emplace(observation_value.value);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

  static std::optional<double> last_value;
};

std::optional<double> TestEvent::last_value{};
PUP::able::PUP_ID TestEvent::my_PUP_ID = 0;  // NOLINT

template <typename Metavariables>
struct Component {
  static constexpr bool local_time_stepping =
      Metavariables::local_time_stepping;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<Tags::Time, Tags::TimeStepId>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<evolution::Actions::RunEventsAndTriggers<
                         local_time_stepping>>>>;
};

template <bool LocalTimeStepping>
struct Metavariables {
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<TestEvent>>,
        tmpl::pair<Trigger, tmpl::push_back<Triggers::logical_triggers,
                                            ::Triggers::OnSubsteps>>>;
  };
};

template <bool LocalTimeStepping>
void test(std::array<
          ActionTesting::MockRuntimeSystem<Metavariables<LocalTimeStepping>>,
          4>& runners) {
  using my_component = Component<Metavariables<LocalTimeStepping>>;

  for (size_t test_case = 0; test_case < 4; ++test_case) {
    auto& runner = runners[test_case];
    ActionTesting::emplace_component<my_component>(&runner, 0);
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  }

  const auto test_all = [&runners](
                            const TimeStepId& id,
                            const std::optional<double> expected_always,
                            const std::optional<double> expected_never,
                            const std::optional<double> expected_always_sub,
                            const std::optional<double> expected_never_sub) {
    const std::array expected{expected_always, expected_never,
                              expected_always_sub, expected_never_sub};
    for (size_t test_case = 0; test_case < 4; ++test_case) {
      auto& runner = runners[test_case];
      auto& box =
          ActionTesting::get_databox<my_component>(make_not_null(&runner), 0);
      db::mutate<Tags::TimeStepId, Tags::Time>(
          [&](const gsl::not_null<TimeStepId*> box_id,
              const gsl::not_null<double*> box_time) {
            *box_id = id;
            *box_time = id.substep_time();
          },
          make_not_null(&box));

      TestEvent::last_value.reset();
      ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
      CHECK(TestEvent::last_value == expected[test_case]);
    }
  };

  const Slab slab(1.0, 2.0);
  const auto start = slab.start();
  const auto step = slab.duration() / 2;
  const auto center = start + step;

  if constexpr (LocalTimeStepping) {
    test_all(TimeStepId(true, 0, start), 1.0, {}, 1.0, {});
    test_all(TimeStepId(true, 0, center), 1.5, {}, 1.5, {});
    test_all(TimeStepId(true, 1, start), 1.0, {}, 1.0, {});
    test_all(TimeStepId(true, 1, center), 1.5, {}, 1.5, {});
    test_all(TimeStepId(true, -1, start), {}, {}, {}, {});
    test_all(TimeStepId(true, -1, center), {}, {}, {}, {});
    test_all(TimeStepId(true, 0, start, 1, step, 1.5), {}, {}, 1000001.0, {});
    test_all(TimeStepId(true, 0, center, 1, step, 2.0), {}, {}, 1000001.5, {});
    test_all(TimeStepId(true, 0, start, 2, step, 1.0), {}, {}, 2000001.0, {});
    test_all(TimeStepId(true, 0, center, 2, step, 1.5), {}, {}, 2000001.5, {});
  } else {
    test_all(TimeStepId(true, 0, start), 1.0, {}, 1.0, {});
    test_all(TimeStepId(true, 0, center), {}, {}, {}, {});
    test_all(TimeStepId(true, 1, start), 1.0, {}, 1.0, {});
    test_all(TimeStepId(true, 1, center), {}, {}, {}, {});
    test_all(TimeStepId(true, -1, start), {}, {}, {}, {});
    test_all(TimeStepId(true, -1, center), {}, {}, {}, {});
    test_all(TimeStepId(true, 0, start, 1, step, 1.5), {}, {}, 1000001.0, {});
    test_all(TimeStepId(true, 0, center, 1, step, 2.0), {}, {}, {}, {});
    test_all(TimeStepId(true, 0, start, 2, step, 1.0), {}, {}, 2000001.0, {});
    test_all(TimeStepId(true, 0, center, 2, step, 1.5), {}, {}, {}, {});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.RunEventsAndTriggers",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables<true>>();
  std::array gts_runners{
      ActionTesting::MockRuntimeSystem<Metavariables<false>>{
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<false>>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - TestEvent\n")},
      ActionTesting::MockRuntimeSystem<Metavariables<false>>{
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<false>>(
              "- Trigger:\n"
              "    Not: Always\n"
              "  Events:\n"
              "    - TestEvent\n")},
      ActionTesting::MockRuntimeSystem<Metavariables<false>>{
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<false>>(
              "- Trigger:\n"
              "    OnSubsteps: Always\n"
              "  Events:\n"
              "    - TestEvent\n")},
      ActionTesting::MockRuntimeSystem<Metavariables<false>>{
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<false>>(
              "- Trigger:\n"
              "    OnSubsteps:\n"
              "      Not: Always\n"
              "  Events:\n"
              "    - TestEvent\n")}};
  test<false>(gts_runners);
  std::array lts_runners{
      ActionTesting::MockRuntimeSystem<Metavariables<true>>{tuples::TaggedTuple<
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>{
          EventsAndTriggers{},
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<true>>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - TestEvent\n")}},
      ActionTesting::MockRuntimeSystem<Metavariables<true>>{tuples::TaggedTuple<
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>{
          EventsAndTriggers{},
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<true>>(
              "- Trigger:\n"
              "    Not: Always\n"
              "  Events:\n"
              "    - TestEvent\n")}},
      ActionTesting::MockRuntimeSystem<Metavariables<true>>{tuples::TaggedTuple<
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>{
          EventsAndTriggers{},
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<true>>(
              "- Trigger:\n"
              "    OnSubsteps: Always\n"
              "  Events:\n"
              "    - TestEvent\n")}},
      ActionTesting::MockRuntimeSystem<Metavariables<true>>{tuples::TaggedTuple<
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
          Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>{
          EventsAndTriggers{},
          TestHelpers::test_creation<EventsAndTriggers, Metavariables<true>>(
              "- Trigger:\n"
              "    OnSubsteps:\n"
              "      Not: Always\n"
              "  Events:\n"
              "    - TestEvent\n")}}};
  test<true>(lts_runners);
}
