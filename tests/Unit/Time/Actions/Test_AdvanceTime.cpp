// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Actions/AdvanceTime.hpp"           // IWYU pragma: keep
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

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
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<TimeStepper>>;

  using simple_tags =
      db::AddSimpleTags<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                        Tags::TimeStep, Tags::Next<Tags::TimeStep>, Tags::Time,
                        Tags::IsUsingTimeSteppingErrorControl,
                        Tags::AdaptiveSteppingDiagnostics>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
                 Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<Actions::AdvanceTime>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

};

void check(std::unique_ptr<TimeStepper> time_stepper,
           const std::vector<Rational>& substeps, const Time& start,
           const TimeDelta& time_step, const bool using_error_control) {
  std::vector<TimeDelta> substep_offsets{};
  substep_offsets.reserve(substeps.size());
  for (const auto& substep : substeps) {
    substep_offsets.push_back(substep * time_step);
  }

  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{std::move(time_stepper)}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeStepId(time_step.is_positive(), 8, start),
       substeps.size() == 1
           ? TimeStepId(time_step.is_positive(), 8, start + time_step)
           : TimeStepId(time_step.is_positive(), 8, start, 1, time_step,
                        (start + substep_offsets[1]).value()),
       time_step, time_step, start.value(), using_error_control,
       AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  for (const auto& step_start : {start, start + time_step}) {
    for (size_t substep = 0; substep < substep_offsets.size(); ++substep) {
      const auto& box = ActionTesting::get_databox<component>(runner, 0);
      const double substep_time =
          (step_start + gsl::at(substep_offsets, substep)).value();
      CHECK(db::get<Tags::TimeStepId>(box) ==
            TimeStepId(time_step.is_positive(), 8, step_start, substep,
                       time_step, substep_time));
      CHECK(db::get<Tags::TimeStep>(box) == time_step);
      CHECK(db::get<Tags::Time>(box) ==
            db::get<Tags::TimeStepId>(box).substep_time());
      runner.next_action<component>(0);
    }
  }

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  const auto& final_time_id = db::get<Tags::TimeStepId>(box);
  const auto expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.step_time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeStepId(time_step.is_positive(), 8, start + 2 * time_step));
  CHECK(db::get<Tags::Time>(box) == final_time_id.substep_time());
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{
            1 + static_cast<uint64_t>(final_time_id.slab_number() - 8), 2, 5, 4,
            5});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.AdvanceTime", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth,
                              TimeSteppers::ClassicalRungeKutta4>();
  const Slab slab(0., 1.);
  check(std::make_unique<TimeSteppers::ClassicalRungeKutta4>(),
        {0, {1, 2}, {1, 2}, 1}, slab.start(), slab.duration() / 2, false);
  check(std::make_unique<TimeSteppers::ClassicalRungeKutta4>(),
        {0, {1, 2}, {1, 2}, 1}, slab.end(), -slab.duration() / 2, false);
  check(std::make_unique<TimeSteppers::AdamsBashforth>(1), {0}, slab.start(),
        slab.duration() / 2, false);
  check(std::make_unique<TimeSteppers::AdamsBashforth>(1), {0}, slab.end(),
        -slab.duration() / 2, false);
  check(std::make_unique<TimeSteppers::ClassicalRungeKutta4>(),
        {0, {1, 2}, {1, 2}, 1, {3, 4}}, slab.start(), slab.duration() / 2,
        true);
  check(std::make_unique<TimeSteppers::ClassicalRungeKutta4>(),
        {0, {1, 2}, {1, 2}, 1, {3, 4}}, slab.end(), -slab.duration() / 2, true);
}
