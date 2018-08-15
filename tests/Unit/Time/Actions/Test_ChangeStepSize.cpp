// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
// IWYU pragma: no_include <pup.h>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
using step_choosers = tmpl::list<StepChoosers::Register::Constant>;
using change_step_size = Actions::ChangeStepSize<step_choosers>;

struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int, tmpl::list<>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = change_step_size::const_global_cache_tags;
};

void check(const bool time_runs_forward, const Time& time, const double request,
           const TimeDelta& expected_step) noexcept {
  CAPTURE(time);
  CAPTURE(request);

  ActionTesting::ActionRunner<Metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(1),
       make_vector<std::unique_ptr<StepChooser<step_choosers>>>(
           std::make_unique<StepChoosers::Constant<step_choosers>>(
               2. * request),
           std::make_unique<StepChoosers::Constant<step_choosers>>(request),
           std::make_unique<StepChoosers::Constant<step_choosers>>(
               2. * request)),
       std::make_unique<StepControllers::BinaryFraction>()}};

  auto box =
      db::create<db::AddSimpleTags<Tags::TimeId, Tags::Next<Tags::TimeId>,
                                   Tags::TimeStep>>(
          TimeId(time_runs_forward, 0, time),
          TimeId(time_runs_forward, 0, time.slab().start()),
          (time_runs_forward ? 1 : -1) * time.slab().duration());
  box = std::get<0>(runner.apply<component, change_step_size>(box, 0));

  CHECK(db::get<Tags::TimeStep>(box) == expected_step);
  CHECK(db::get<Tags::Next<Tags::TimeId>>(box) ==
        TimeId(time_runs_forward, 0, time + expected_step));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeStepSize", "[Unit][Time][Actions]") {
  const Slab slab(-5., -2.);
  const double slab_length = slab.duration().value();
  check(true, slab.start() + slab.duration() / 4, slab_length / 5.,
        slab.duration() / 8);
  check(true, slab.start() + slab.duration() / 4, slab_length,
        slab.duration() / 4);
  check(false, slab.end() - slab.duration() / 4, slab_length / 5.,
        -slab.duration() / 8);
  check(false, slab.end() - slab.duration() / 4, slab_length,
        -slab.duration() / 4);
}
