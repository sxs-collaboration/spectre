// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
namespace Tags {
template <typename Tag>
struct Next;
template <typename Tag>
struct dt;
}  // namespace Tags

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct Component;

struct Metavariables {
  using component_list = tmpl::list<Component>;
};

struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using const_global_cache_tags =
      tmpl::list<Tags::ConcreteTimeStepper<TimeStepper>>;

  using simple_tags =
      tmpl::list<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 Tags::Next<Tags::TimeStep>, Tags::HistoryEvolvedVariables<Var>,
                 Tags::AdaptiveSteppingDiagnostics,
                 ::Tags::ChangeSlabSize::SlabSizeGoal>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::ChangeSlabSize>>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeSlabSize", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::Rk3HesthavenSsp>();

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::make_unique<TimeSteppers::Rk3HesthavenSsp>()}};

  ActionTesting::emplace_component_and_initialize<Component>(&runner, 0, {});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& box = ActionTesting::get_databox<Component>(make_not_null(&runner), 0);

  for (const bool time_runs_forward : {true, false}) {
    const double unit_step = time_runs_forward ? 1.0 : -1.0;
    Slab slab(1.5, 2.0);
    Time start_time;

    const auto resize_slab = [&slab, &start_time,
                              &time_runs_forward](const double length) {
      if (time_runs_forward) {
        slab = slab.with_duration_from_start(length);
        start_time = slab.start();
      } else {
        slab = slab.with_duration_to_end(length);
        start_time = slab.end();
      }
    };

    resize_slab(slab.duration().value());

    const auto get_step = [](const TimeStepId& id) {
      return (id.time_runs_forward() ? 1 : -1) *
             id.step_time().slab().duration() / 2;
    };

    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::Next<Tags::TimeStep>, Tags::AdaptiveSteppingDiagnostics,
               ::Tags::ChangeSlabSize::SlabSizeGoal>(
        [&get_step, &start_time, &time_runs_forward](
            const gsl::not_null<TimeStepId*> id,
            const gsl::not_null<TimeStepId*> next_id,
            const gsl::not_null<TimeDelta*> step,
            const gsl::not_null<TimeDelta*> next_step,
            const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
            const gsl::not_null<double*> slab_size_goal,
            const TimeStepper& stepper) {
          *id = TimeStepId(time_runs_forward, 3, start_time);
          *step = get_step(*id);
          *next_step = get_step(*id);
          *next_id = stepper.next_time_id(*id, *step);
          *diags = AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5};
          // Step is slab/2 for test
          *slab_size_goal = (2 * *step).value();
        },
        make_not_null(&box), db::get<Tags::TimeStepper<TimeStepper>>(box));

    using ExpectedMessages = ::Tags::ChangeSlabSize::NumberOfExpectedMessages;
    using NewSize = ::Tags::ChangeSlabSize::NewSlabSize;

    const auto check_box = [&box, &get_step](const TimeStepId& id,
                                             const uint64_t changes,
                                             const double expected_goal) {
      CHECK(db::get<Tags::TimeStepId>(box) == id);
      CHECK(db::get<Tags::TimeStep>(box) == get_step(id));
      CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) ==
            db::get<Tags::TimeStepper<TimeStepper>>(box).next_time_id(
                db::get<Tags::TimeStepId>(box), db::get<Tags::TimeStep>(box)));
      CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
            AdaptiveSteppingDiagnostics{1, 2 + changes, 3, 4, 5});
      CHECK(db::get<::Tags::ChangeSlabSize::SlabSizeGoal>(box) ==
            expected_goal);
    };

    // Nothing to do
    {
      runner.next_action<Component>(0);
      check_box(TimeStepId(time_runs_forward, 3, start_time), 0,
                ((time_runs_forward ? 1 : -1) * slab.duration()).value());
    }

    // Nothing to do, but not at the goal.
    {
      db::mutate<::Tags::ChangeSlabSize::SlabSizeGoal>(
          [&unit_step](const gsl::not_null<double*> slab_size_goal) {
            *slab_size_goal = 2.0 * unit_step;
          },
          make_not_null(&box));
      runner.next_action<Component>(0);
      resize_slab(2.0);
      check_box(TimeStepId(time_runs_forward, 3, start_time), 1,
                2.0 * unit_step);
    }

    // Simple case
    {
      db::mutate<ExpectedMessages>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
            ++(*expected)[3];
          },
          make_not_null(&box));
      REQUIRE_FALSE(ActionTesting::next_action_if_ready<Component>(
          make_not_null(&runner), 0));
      db::mutate<NewSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            (*sizes)[3]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size = unit_step});
          },
          make_not_null(&box));

      runner.next_action<Component>(0);
      resize_slab(1.0);
      check_box(TimeStepId(time_runs_forward, 3, start_time), 2,
                2.0 * unit_step);
      CHECK(db::get<ExpectedMessages>(box).empty());
      CHECK(db::get<NewSize>(box).empty());
      db::mutate<ExpectedMessages, NewSize>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected,
              const gsl::not_null<
                  std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            expected->clear();
            sizes->clear();
          },
          make_not_null(&box));
    }

    // Multiple messages at multiple times
    {
      db::mutate<ExpectedMessages>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
            (*expected)[3] += 2;
            (*expected)[4] += 1;
          },
          make_not_null(&box));
      REQUIRE_FALSE(ActionTesting::next_action_if_ready<Component>(
          make_not_null(&runner), 0));
      db::mutate<NewSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            (*sizes)[3]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 2.0 * unit_step});
          },
          make_not_null(&box));
      REQUIRE_FALSE(ActionTesting::next_action_if_ready<Component>(
          make_not_null(&runner), 0));
      db::mutate<NewSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            (*sizes)[4]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 0.5 * unit_step});
          },
          make_not_null(&box));
      REQUIRE_FALSE(ActionTesting::next_action_if_ready<Component>(
          make_not_null(&runner), 0));
      db::mutate<ExpectedMessages>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
            ++(*expected)[4];
          },
          make_not_null(&box));
      db::mutate<NewSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            (*sizes)[3]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 3.0 * unit_step});
          },
          make_not_null(&box));
      runner.next_action<Component>(0);
      resize_slab(2.0);
      check_box(TimeStepId(time_runs_forward, 3, start_time), 3,
                2.0 * unit_step);
      CHECK(db::get<ExpectedMessages>(box).size() == 1);
      CHECK(db::get<ExpectedMessages>(box).count(4) == 1);
      CHECK(db::get<NewSize>(box).size() == 1);
      CHECK(db::get<NewSize>(box).count(4) == 1);
      db::mutate<ExpectedMessages, NewSize>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected,
              const gsl::not_null<
                  std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            expected->clear();
            sizes->clear();
          },
          make_not_null(&box));
    }

    // Check interior of slab
    {
      db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>>(
          [&start_time, &time_runs_forward](
              const gsl::not_null<TimeStepId*> id,
              const gsl::not_null<TimeStepId*> next_id, const TimeDelta& step,
              const TimeStepper& stepper) {
            *id = TimeStepId(time_runs_forward, 3, start_time + step);
            *next_id = stepper.next_time_id(*id, step);
          },
          make_not_null(&box), db::get<Tags::TimeStep>(box),
          db::get<Tags::TimeStepper<TimeStepper>>(box));
      const TimeStepId initial_id = db::get<Tags::TimeStepId>(box);
      db::mutate<ExpectedMessages>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
            ++(*expected)[3];
            ++(*expected)[4];
          },
          make_not_null(&box));
      runner.next_action<Component>(0);
      check_box(initial_id, 3, 2.0 * unit_step);
      CHECK(db::get<ExpectedMessages>(box).size() == 2);
      CHECK(db::get<ExpectedMessages>(box).count(3) == 1);
      CHECK(db::get<ExpectedMessages>(box).count(4) == 1);
      CHECK(db::get<NewSize>(box).empty());
      db::mutate<NewSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            (*sizes)[3]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 0.1 * unit_step});
            (*sizes)[4]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 0.1 * unit_step});
          },
          make_not_null(&box));
      runner.next_action<Component>(0);
      check_box(initial_id, 3, 2.0 * unit_step);
      db::mutate<ExpectedMessages, NewSize>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected,
              const gsl::not_null<
                  std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            expected->clear();
            sizes->clear();
          },
          make_not_null(&box));
    }

    // Check at a substep
    {
      db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>>(
          [](const gsl::not_null<TimeStepId*> id,
             const gsl::not_null<TimeStepId*> next_id, const TimeDelta& step,
             const TimeStepper& stepper) {
            const auto local_slab = id->step_time().slab();
            while (id->substep_time() != local_slab.start().value() and
                   id->substep_time() != local_slab.end().value()) {
              *id = *next_id;
              REQUIRE(id->substep() != 0);
              *next_id = stepper.next_time_id(*id, step);
            }
          },
          make_not_null(&box), db::get<Tags::TimeStep>(box),
          db::get<Tags::TimeStepper<TimeStepper>>(box));
      const TimeStepId initial_id = db::get<Tags::TimeStepId>(box);
      db::mutate<ExpectedMessages>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
            ++(*expected)[3];
            ++(*expected)[4];
          },
          make_not_null(&box));
      runner.next_action<Component>(0);
      check_box(initial_id, 3, 2.0 * unit_step);
      CHECK(db::get<ExpectedMessages>(box).size() == 2);
      CHECK(db::get<ExpectedMessages>(box).count(3) == 1);
      CHECK(db::get<ExpectedMessages>(box).count(4) == 1);
      CHECK(db::get<NewSize>(box).empty());
      db::mutate<NewSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            (*sizes)[3]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 0.1 * unit_step});
            (*sizes)[4]
                .emplace_back(time_runs_forward)
                .process(TimeStepRequest{.size_goal = 0.1 * unit_step});
          },
          make_not_null(&box));
      runner.next_action<Component>(0);
      check_box(initial_id, 3, 2.0 * unit_step);
      db::mutate<ExpectedMessages, NewSize>(
          [&](const gsl::not_null<std::map<int64_t, size_t>*> expected,
              const gsl::not_null<
                  std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) {
            expected->clear();
            sizes->clear();
          },
          make_not_null(&box));
    }
  }
}
