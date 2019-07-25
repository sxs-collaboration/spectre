// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Actions/AdvanceTime.hpp"           // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <initializer_list>
// IWYU pragma: no_include <unordered_map>

class TimeStepper;
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare db::DataBox

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;

  using simple_tags =
      db::AddSimpleTags<Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::AdvanceTime>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialization, Testing, Exit };
};

void check_rk3(const Time& start, const TimeDelta& time_step) {
  const std::array<TimeDelta, 3> substep_offsets{
      {time_step * 0, time_step, time_step / 2}};

  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::RungeKutta3>()}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeId(time_step.is_positive(), 8, start),
       TimeId(time_step.is_positive(), 8, start, 1, start + substep_offsets[1]),
       time_step});
  runner.set_phase(Metavariables::Phase::Testing);

  for (const auto& step_start : {start, start + time_step}) {
    for (size_t substep = 0; substep < 3; ++substep) {
      const auto& box =
          ActionTesting::get_databox<component,
                                     typename component::simple_tags>(runner,
                                                                      0);
      CHECK(db::get<Tags::TimeId>(box) ==
            TimeId(time_step.is_positive(), 8, step_start, substep,
                   step_start + gsl::at(substep_offsets, substep)));
      CHECK(db::get<Tags::TimeStep>(box) == time_step);
      runner.next_action<component>(0);
    }
  }

  const auto& box =
      ActionTesting::get_databox<component, typename component::simple_tags>(
          runner, 0);
  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeId(time_step.is_positive(), 8, start + 2 * time_step));
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
}

void check_abn(const Time& start, const TimeDelta& time_step) {
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeId(time_step.is_positive(), 8, start),
       TimeId(time_step.is_positive(), 8, start + time_step), time_step});
  runner.set_phase(Metavariables::Phase::Testing);

  for (const auto& step_start : {start, start + time_step}) {
    const auto& box =
        ActionTesting::get_databox<component, typename component::simple_tags>(
            runner, 0);
    CHECK(db::get<Tags::TimeId>(box) ==
          TimeId(time_step.is_positive(), 8, step_start));
    CHECK(db::get<Tags::TimeStep>(box) == time_step);
    runner.next_action<component>(0);
  }

  const auto& box =
      ActionTesting::get_databox<component, typename component::simple_tags>(
          runner, 0);
  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeId(time_step.is_positive(), 8, start + 2 * time_step));
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.AdvanceTime", "[Unit][Time][Actions]") {
  const Slab slab(0., 1.);
  check_rk3(slab.start(), slab.duration() / 2);
  check_rk3(slab.end(), -slab.duration() / 2);
  check_abn(slab.start(), slab.duration() / 2);
  check_abn(slab.end(), -slab.duration() / 2);
}
