// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class LtsTimeStepper;
class TimeStepper;
/// \endcond

namespace TimeStepperTestUtils {

void check_multistep_properties(const TimeStepper& stepper);

void check_substep_properties(const TimeStepper& stepper);

void integrate_test(const TimeStepper& stepper, size_t order,
                    size_t number_of_past_steps, double integration_time,
                    double epsilon, bool test_apply_twice = false);

void integrate_test_explicit_time_dependence(const TimeStepper& stepper,
                                             size_t order,
                                             size_t number_of_past_steps,
                                             double integration_time,
                                             double epsilon);

void integrate_variable_test(const TimeStepper& stepper, size_t order,
                             size_t number_of_past_steps, double epsilon);

void integrate_error_test(const TimeStepper& stepper, size_t order,
                          size_t number_of_past_steps, double integration_time,
                          double epsilon, size_t num_steps, double error_factor,
                          bool test_apply_twice = false);

template <typename F1, typename F2, typename EvolvedType>
void initialize_history(
    Time time,
    const gsl::not_null<TimeSteppers::History<EvolvedType, EvolvedType>*>
        history,
    F1&& analytic, F2&& rhs, TimeDelta step_size,
    const size_t number_of_past_steps) {
  int64_t slab_number = -1;
  for (size_t j = 0; j < number_of_past_steps; ++j) {
    ASSERT(time.slab() == step_size.slab(), "Slab mismatch");
    if ((step_size.is_positive() and time.is_at_slab_start()) or
        (not step_size.is_positive() and time.is_at_slab_end())) {
      const Slab new_slab = time.slab().advance_towards(-step_size);
      time = time.with_slab(new_slab);
      step_size = step_size.with_slab(new_slab);
      --slab_number;
    }
    time -= step_size;
    history->insert_initial(
        TimeStepId(step_size.is_positive(), slab_number, time),
        rhs(analytic(time.value()), time.value()));
    if (j == 0) {
      history->most_recent_value() = analytic(time.value());
    }
  }
}

void stability_test(const TimeStepper& stepper);

void equal_rate_boundary(const LtsTimeStepper& stepper, size_t order,
                         size_t number_of_past_steps, double epsilon,
                         bool forward);

void check_convergence_order(const TimeStepper& stepper);

void check_dense_output(const TimeStepper& stepper,
                        const size_t history_integration_order);

void check_boundary_dense_output(const LtsTimeStepper& stepper);
}  // namespace TimeStepperTestUtils
