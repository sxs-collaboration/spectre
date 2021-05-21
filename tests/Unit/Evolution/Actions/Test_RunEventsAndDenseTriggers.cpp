// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/History.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct PrimVar : db::SimpleTag {
  using type = double;
};

class TestTrigger : public DenseTrigger {
 public:
  TestTrigger() = default;
  explicit TestTrigger(CkMigrateMessage* const msg) noexcept
      : DenseTrigger(msg) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestTrigger);  // NOLINT
#pragma GCC diagnostic pop

  // All triggers are evaluated once at the start of the evolution, so
  // we have to handle that call and set up for triggering at the
  // interesting time.
  TestTrigger(const double init_time, const double trigger_time,
              const bool is_ready, const bool is_triggered) noexcept
      : init_time_(init_time),
        trigger_time_(trigger_time),
        is_ready_(is_ready),
        is_triggered_(is_triggered) {}

  using is_triggered_argument_tags = tmpl::list<Tags::Time>;
  Result is_triggered(const double time) const noexcept {
    if (time == init_time_) {
      return {false, trigger_time_};
    }
    CHECK(time == trigger_time_);
    return {is_triggered_, (trigger_time_ > init_time_ ? 1.0 : -1.0) *
                               std::numeric_limits<double>::infinity()};
  }

  using is_ready_argument_tags = tmpl::list<Tags::Time>;
  bool is_ready(const double time) const noexcept {
    if (time == init_time_) {
      return true;
    }
    CHECK(time == trigger_time_);
    return is_ready_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    DenseTrigger::pup(p);
    p | init_time_;
    p | trigger_time_;
    p | is_ready_;
    p | is_triggered_;
  }

 private:
  double init_time_ = std::numeric_limits<double>::signaling_NaN();
  double trigger_time_ = std::numeric_limits<double>::signaling_NaN();
  bool is_ready_ = false;
  bool is_triggered_ = false;
};

PUP::able::PUP_ID TestTrigger::my_PUP_ID = 0;  // NOLINT

struct TestEvent : public Event {
  TestEvent() = default;
  explicit TestEvent(CkMigrateMessage* const /*msg*/) noexcept {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestEvent);  // NOLINT
#pragma GCC diagnostic pop

  explicit TestEvent(const bool needs_evolved_variables)
      : needs_evolved_variables_(needs_evolved_variables) {}

  using argument_tags = tmpl::list<Tags::Time, Var, PrimVar>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(const double time, const double var, const double prim_var,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/) const noexcept {
    calls.emplace_back(time, var, prim_var);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const noexcept {
    // We use the triggers to control readiness for this test.
    return true;
  }

  bool needs_evolved_variables() const noexcept override {
    return needs_evolved_variables_;
  }

  bool needs_evolved_variables_ = false;

  static std::vector<std::tuple<double, double, double>> calls;
};

std::vector<std::tuple<double, double, double>> TestEvent::calls{};

PUP::able::PUP_ID TestEvent::my_PUP_ID = 0;  // NOLINT

struct PrimFromCon {
  using return_tags = tmpl::list<PrimVar>;
  using argument_tags = tmpl::list<Var>;
  static void apply(const gsl::not_null<double*> prim,
                    const double con) noexcept {
    *prim = -con;
  }
};

template <bool HasPrimitiveAndConservativeVars>
struct System {
  using variables_tag = Var;
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveAndConservativeVars;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using initialization_tags =
      tmpl::list<Tags::TimeStepId, Tags::TimeStep, Tags::Time, Var, PrimVar,
                 Tags::HistoryEvolvedVariables<Var>,
                 evolution::Tags::EventsAndDenseTriggers>;

  using prim_from_con = tmpl::conditional_t<
      metavariables::system::has_primitive_and_conservative_vars, PrimFromCon,
      void>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<
          evolution::Actions::RunEventsAndDenseTriggers<prim_from_con>>>>;
};

template <bool HasPrimitiveAndConservativeVars>
struct Metavariables {
  using system = System<HasPrimitiveAndConservativeVars>;
  static constexpr bool local_time_stepping = false;
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<TimeStepper>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DenseTrigger, tmpl::list<TestTrigger>>,
                  tmpl::pair<Event, tmpl::list<TestEvent>>>;
  };
  enum class Phase { Initialization, Testing, Exit };
};

template <bool HasPrimitiveAndConservativeVars>
void test(const bool time_runs_forward) noexcept {
  using metavars = Metavariables<HasPrimitiveAndConservativeVars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using component = Component<metavars>;

  const Slab slab(0.0, 4.0);
  const TimeStepId time_step_id(time_runs_forward, 0,
                                slab.start() + slab.duration() / 2);
  const TimeDelta exact_step_size =
      (time_runs_forward ? 1 : -1) * slab.duration() / 4;
  const double start_time = time_step_id.step_time().value();
  const double step_size = exact_step_size.value();
  const double step_center = start_time + 0.5 * step_size;
  const double initial_var = 8.0;

  const auto set_up_component =
      [&exact_step_size, &initial_var, &start_time, &time_step_id](
          const gsl::not_null<MockRuntimeSystem*> runner,
          const std::vector<std::tuple<double, bool, bool, bool>>&
              triggers) noexcept {
        TimeSteppers::History<double, double> history(1);
        history.insert(time_step_id, 1.0);
        history.most_recent_value() = initial_var;

        evolution::EventsAndDenseTriggers::ConstructionType
            events_and_dense_triggers{};
        for (auto [trigger_time, is_ready, is_triggered,
                   needs_evolved_variables] : triggers) {
          events_and_dense_triggers.emplace(
              std::make_unique<TestTrigger>(start_time, trigger_time, is_ready,
                                            is_triggered),
              make_vector<std::unique_ptr<Event>>(
                  std::make_unique<TestEvent>(needs_evolved_variables)));
        }

        ActionTesting::emplace_array_component<component>(
            runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
            time_step_id, exact_step_size, start_time, initial_var, 0.0,
            std::move(history),
            evolution::EventsAndDenseTriggers(
                std::move(events_and_dense_triggers)));
        ActionTesting::set_phase(runner, metavars::Phase::Testing);
      };

  const auto check_event_calls =
      [](const std::vector<std::tuple<double, double, double>>&
             expected) noexcept {
        CAPTURE(get_output(expected));
        CAPTURE(get_output(TestEvent::calls));
        REQUIRE(TestEvent::calls.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
          CHECK(std::get<0>(TestEvent::calls[i]) == std::get<0>(expected[i]));
          CHECK(std::get<1>(TestEvent::calls[i]) == std::get<1>(expected[i]));
          if (HasPrimitiveAndConservativeVars) {
            CHECK(std::get<2>(TestEvent::calls[i]) == std::get<2>(expected[i]));
          }
        }
        TestEvent::calls.clear();
      };

  // Tests start here

  // Nothing should happen in self-start
  const auto check_self_start = [&check_event_calls, &set_up_component,
                                 &start_time, &step_size](
                                    const bool trigger_is_ready) noexcept {
    // This isn't a valid time for the trigger to reschedule to (it is
    // in the past), but the triggers should be completely ignored in
    // this check.
    const double invalid_time = start_time - step_size;

    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(
        &runner, {{invalid_time, trigger_is_ready, trigger_is_ready, false}});
    {
      auto& box = ActionTesting::get_databox<component, tmpl::list<>>(
          make_not_null(&runner), 0);
      db::mutate<Tags::TimeStepId>(
          make_not_null(&box),
          [](const gsl::not_null<TimeStepId*> id) noexcept {
            *id = TimeStepId(id->time_runs_forward(), -1, id->step_time());
          });
    }
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
    check_event_calls({});
  };
  check_self_start(true);
  check_self_start(false);

  // No triggers
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {});
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
  }

  // Triggers too far in the future
  const auto check_not_reached = [&check_event_calls, &set_up_component,
                                  &start_time, &step_size](
                                     const bool trigger_is_ready) noexcept {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{start_time + 1.5 * step_size, trigger_is_ready,
                                trigger_is_ready, false}});
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
    check_event_calls({});
  };
  check_not_reached(true);
  check_not_reached(false);

  // Trigger isn't ready
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, false, true, false}});
    CHECK(not ActionTesting::next_action_if_ready<component>(
        make_not_null(&runner), 0));
    check_event_calls({});
  }

  // Variables not needed
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, true, true, false}});
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
    check_event_calls({{step_center, initial_var, 0.0}});
  }

  // Variables needed
  {
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, true, true, true}});
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
    const double dense_var = initial_var + 0.5 * step_size;
    check_event_calls({{step_center, dense_var, -dense_var}});
  }

  // Missing dense output data
  const auto check_missing_dense_data = [&check_event_calls, &initial_var,
                                         &set_up_component, &step_center,
                                         &time_step_id](
                                            const bool data_needed) noexcept {
    MockRuntimeSystem runner{{std::make_unique<TimeSteppers::RungeKutta3>()}};
    set_up_component(&runner, {{step_center, true, true, data_needed}});
    {
      auto& box = ActionTesting::get_databox<component, tmpl::list<>>(
          make_not_null(&runner), 0);
      db::mutate<Tags::HistoryEvolvedVariables<Var>>(
          make_not_null(&box),
          [&initial_var, &time_step_id](
              const gsl::not_null<TimeSteppers::History<double, double>*>
                  history) noexcept {
            *history = TimeSteppers::History<double, double>(3);
            history->insert(TimeStepId(time_step_id.time_runs_forward(), 0,
                                       time_step_id.step_time(), 1,
                                       time_step_id.step_time()),
                            1.0);
            history->most_recent_value() = initial_var;
          });
    }
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
    if (data_needed) {
      check_event_calls({});
    } else {
      // If we don't need the data, it shouldn't matter whether it is missing.
      check_event_calls({{step_center, initial_var, 0.0}});
    }
  };
  check_missing_dense_data(true);
  check_missing_dense_data(false);

  // Multiple triggers
  {
    const double second_trigger = start_time + 0.75 * step_size;
    MockRuntimeSystem runner{
        {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
    set_up_component(&runner, {{step_center, true, true, false},
                               {second_trigger, true, true, false}});
    CHECK(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         0));
    check_event_calls(
        {{step_center, initial_var, 0.0}, {second_trigger, initial_var, 0.0}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.RunEventsAndDenseTriggers",
                  "[Unit][Evolution][Actions]") {
  Parallel::register_classes_with_charm<TimeSteppers::AdamsBashforthN,
                                        TimeSteppers::RungeKutta3>();
  // Same lists for true and false
  Parallel::register_factory_classes_with_charm<Metavariables<true>>();
  test<false>(true);
  test<false>(false);
  test<true>(true);
  test<true>(false);
}
