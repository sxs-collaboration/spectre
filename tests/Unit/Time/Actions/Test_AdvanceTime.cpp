// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
// IWYU pragma: no_include <initializer_list>
#include <memory>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int,
                                      tmpl::list<CacheTags::TimeStepper>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};

void check_rk3(const Time& start, const TimeDelta& time_step) {
  ActionTesting::ActionRunner<Metavariables> runner{
      {std::make_unique<TimeSteppers::RungeKutta3>()}};

  const std::array<TimeDelta, 3> substep_offsets{
      {time_step * 0, time_step, time_step / 2}};

  auto box = db::create<db::AddSimpleTags<
      Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>>(
      TimeId(time_step.is_positive(), 8, start),
      TimeId(time_step.is_positive(), 8, start, 1, start + substep_offsets[1]),
      time_step);

  for (const auto& step_start : {start, start + time_step}) {
    for (size_t substep = 0; substep < 3; ++substep) {
      CHECK(db::get<Tags::TimeId>(box) ==
            TimeId(time_step.is_positive(), 8, step_start, substep,
                   step_start + gsl::at(substep_offsets, substep)));
      CHECK(db::get<Tags::TimeStep>(box) == time_step);
      box = std::get<0>(runner.apply<component, Actions::AdvanceTime>(box, 0));
    }
  }

  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto& expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeId(time_step.is_positive(), 9, start + 2 * time_step));
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
}

void check_abn(const Time& start, const TimeDelta& time_step) {
  ActionTesting::ActionRunner<Metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(1)}};

  auto box =
      db::create<db::AddSimpleTags<Tags::TimeId, Tags::Next<Tags::TimeId>,
                                   Tags::TimeStep>>(
          TimeId(time_step.is_positive(), 8, start),
          TimeId(time_step.is_positive(), 8, start + time_step), time_step);

  for (const auto& step_start : {start, start + time_step}) {
    CHECK(db::get<Tags::TimeId>(box) ==
          TimeId(time_step.is_positive(), 8, step_start));
    CHECK(db::get<Tags::TimeStep>(box) == time_step);
    box = std::get<0>(runner.apply<component, Actions::AdvanceTime>(box, 0));
  }

  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto& expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time().slab() == expected_slab);
  CHECK(final_time_id ==
        TimeId(time_step.is_positive(), 9, start + 2 * time_step));
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
