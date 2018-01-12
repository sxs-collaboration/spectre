// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int,
                                      tmpl::list<CacheTags::TimeStepper>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
};

void do_check(const Time& start, const TimeDelta& time_step) {
  ActionTesting::ActionRunner<Metavariables> runner{
    {std::make_unique<TimeSteppers::RungeKutta3>()}};

  auto box = db::create<db::AddTags<Tags::TimeId, Tags::TimeStep>>(
      TimeId{8, start, 0}, time_step);

  std::array<TimeDelta, 3> substep_offsets{
    {time_step * 0, time_step, time_step / 2}};

  for (const auto& step_start : {start, start + time_step}) {
    for (size_t substep = 0; substep < 3; ++substep) {
      CHECK(db::get<Tags::TimeId>(box) ==
            (TimeId{
              8, step_start + gsl::at(substep_offsets, substep), substep}));
      CHECK(db::get<Tags::TimeStep>(box) == time_step);
      box = std::get<0>(runner.apply<component, Actions::AdvanceTime>(box, 0));
    }
  }

  const auto& final_time_id = db::get<Tags::TimeId>(box);
  const auto& expected_slab = start.slab().advance_towards(time_step);
  CHECK(final_time_id.time.slab() == expected_slab);
  CHECK(final_time_id == (TimeId{9, start + 2 * time_step, 0}));
  CHECK(db::get<Tags::TimeStep>(box) == time_step.with_slab(expected_slab));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.AdvanceTime", "[Unit][Time][Actions]") {
  const Slab slab(0., 1.);
  do_check(slab.start(), slab.duration() / 2);
  do_check(slab.end(), -slab.duration() / 2);
}
