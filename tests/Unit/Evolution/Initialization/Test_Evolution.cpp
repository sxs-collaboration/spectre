// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct TestMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        StepChooser<StepChooserUse::LtsStep>,
        tmpl::list<StepChoosers::Increase<StepChooserUse::LtsStep>>>>;
  };
};

void test_gts() {
  const double initial_time = 1.5;
  const double initial_dt = 0.5;
  const double initial_slab_size = initial_dt;
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::AdamsBashforth>(5);

  const Slab initial_slab =
      Slab::with_duration_from_start(initial_time, initial_slab_size);
  const Time time = initial_slab.start();
  const TimeStepId expected_next_time_step_id = TimeStepId(
      true, -static_cast<int64_t>(time_stepper->number_of_past_steps()), time);
  const TimeDelta expected_time_step = time.slab().duration();
  const TimeDelta expected_next_time_step = expected_time_step;

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Time, Initialization::Tags::InitialTimeDelta,
      Initialization::Tags::InitialSlabSize<false>,
      ::Tags::TimeStepper<TimeStepper>, ::Tags::Next<::Tags::TimeStepId>,
      ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>>>(
      initial_time, initial_dt, initial_slab_size, std::move(time_stepper),
      TimeStepId{}, TimeDelta{}, TimeDelta{});

  db::mutate_apply<Initialization::TimeStepping<TestMetavariables, false>>(
      make_not_null(&box));

  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::Next<::Tags::TimeStep>>(box) ==
        expected_next_time_step);
}

void test_lts() {
  const double initial_time = 1.5;
  const double initial_dt = 0.5;
  const double initial_slab_size = 4.5;
  std::unique_ptr<LtsTimeStepper> lts_time_stepper =
      std::make_unique<TimeSteppers::AdamsBashforth>(5);

  const Slab initial_slab =
      Slab::with_duration_from_start(initial_time, initial_slab_size);
  const Time time = initial_slab.start();
  const TimeStepId expected_next_time_step_id = TimeStepId(
      true, -static_cast<int64_t>(lts_time_stepper->number_of_past_steps()),
      time);
  const TimeDelta expected_time_step = choose_lts_step_size(time, initial_dt);
  const TimeDelta expected_next_time_step = expected_time_step;

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Time, Initialization::Tags::InitialTimeDelta,
      Initialization::Tags::InitialSlabSize<true>,
      ::Tags::TimeStepper<LtsTimeStepper>, ::Tags::Next<::Tags::TimeStepId>,
      ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>>>(
      initial_time, initial_dt, initial_slab_size, std::move(lts_time_stepper),
      TimeStepId{}, TimeDelta{}, TimeDelta{});

  db::mutate_apply<Initialization::TimeStepping<TestMetavariables, true>>(
      make_not_null(&box));

  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::Next<::Tags::TimeStep>>(box) ==
        expected_next_time_step);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.TimeStepping",
                  "[Evolution][Unit]") {
  test_gts();
  test_lts();
}
