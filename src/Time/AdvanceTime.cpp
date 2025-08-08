// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/AdvanceTime.hpp"

#include <cstdint>

#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"

void AdvanceTime::apply(const gsl::not_null<TimeStepId*> time_id,
                        const gsl::not_null<TimeStepId*> next_time_id,
                        const gsl::not_null<TimeDelta*> time_step,
                        const gsl::not_null<double*> time,
                        const gsl::not_null<uint64_t*> step_number_within_slab,
                        const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
                        const TimeStepper& time_stepper,
                        const bool using_error_control) {
  const bool new_step = next_time_id->substep() == 0;
  if (time_id->slab_number() != next_time_id->slab_number()) {
    *step_number_within_slab = 0;
    ++diags->number_of_slabs;
    // Put this here instead of unconditionally doing the next
    // check because on the first call time_id doesn't have a
    // valid slab so comparing the times will FPE.
    ++diags->number_of_steps;
  } else if (new_step) {
    ++(*step_number_within_slab);
    ++diags->number_of_steps;
  }

  *time_id = *next_time_id;
  *time_step = time_step->with_slab(time_id->step_time().slab());

  if (using_error_control) {
    *next_time_id =
        time_stepper.next_time_id_for_error(*next_time_id, *time_step);
  } else {
    *next_time_id = time_stepper.next_time_id(*next_time_id, *time_step);
  }
  *time = time_id->substep_time();
}
