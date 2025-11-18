// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Rational.hpp"

class TimeStepper;

namespace {
void check(std::unique_ptr<TimeStepper> time_stepper,
           const std::vector<Rational>& substeps, const Time& start,
           const TimeDelta& time_step, const bool using_error_control) {
  std::vector<TimeDelta> substep_offsets{};
  substep_offsets.reserve(substeps.size());
  for (const auto& substep : substeps) {
    substep_offsets.push_back(substep * time_step);
  }

  auto box = db::create<
      db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>,
                        Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                        Tags::TimeStep, Tags::Time, Tags::StepNumberWithinSlab,
                        Tags::StepperErrorEstimatesEnabled,
                        Tags::AdaptiveSteppingDiagnostics>,
      time_stepper_ref_tags<TimeStepper>>(
      std::move(time_stepper), TimeStepId(time_step.is_positive(), 8, start),
      substeps.size() == 1
          ? TimeStepId(time_step.is_positive(), 8, start + time_step)
          : TimeStepId(time_step.is_positive(), 8, start, 1, time_step,
                       (start + substep_offsets[1]).value()),
      time_step, start.value(), uint64_t{0}, using_error_control,
      AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  uint64_t step_number_within_slab = 0;
  auto current_slab_number =
      db::get<Tags::AdaptiveSteppingDiagnostics>(box).number_of_slabs;
  for (const auto& step_start : {start, start + time_step}) {
    for (size_t substep = 0; substep < substep_offsets.size(); ++substep) {
      const double substep_time =
          (step_start + gsl::at(substep_offsets, substep)).value();
      CHECK(db::get<Tags::TimeStepId>(box) ==
            TimeStepId(time_step.is_positive(), 8, step_start, substep,
                       time_step, substep_time));
      CHECK(db::get<Tags::TimeStep>(box) == time_step);
      CHECK(db::get<Tags::Time>(box) ==
            db::get<Tags::TimeStepId>(box).substep_time());
      CHECK(db::get<Tags::StepNumberWithinSlab>(box) ==
            step_number_within_slab);
      db::mutate_apply<AdvanceTime>(make_not_null(&box));
    }
    if (current_slab_number ==
        db::get<Tags::AdaptiveSteppingDiagnostics>(box).number_of_slabs) {
      ++step_number_within_slab;
    } else {
      step_number_within_slab = 0;
      ++current_slab_number;
      ASSERT(
          current_slab_number ==
              db::get<Tags::AdaptiveSteppingDiagnostics>(box).number_of_slabs,
          "Current slab number is not what I expected");
    }
  }

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

SPECTRE_TEST_CASE("Unit.Time.AdvanceTime", "[Unit][Time]") {
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
}  // namespace
