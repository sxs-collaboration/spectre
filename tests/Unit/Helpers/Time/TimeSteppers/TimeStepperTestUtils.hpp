// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

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
                    double epsilon);

void integrate_test_explicit_time_dependence(const TimeStepper& stepper,
                                             size_t order,
                                             size_t number_of_past_steps,
                                             double integration_time,
                                             double epsilon);

void integrate_variable_test(const TimeStepper& stepper, size_t order,
                             size_t number_of_past_steps, double epsilon);

void integrate_error_test(const TimeStepper& stepper, size_t order,
                          size_t number_of_past_steps, double integration_time,
                          double epsilon, size_t num_steps,
                          double error_factor);

template <typename F1, typename F2, typename EvolvedType>
void initialize_history(
    Time time,
    const gsl::not_null<TimeSteppers::History<EvolvedType>*> history,
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
        analytic(time.value()), rhs(analytic(time.value()), time.value()));
  }
}

/// Check that the reported stable step is correct.  The \p phase
/// argument is the phase that is most unstable in the differential
/// equation \f$y' = (\exp(i \phi) - 1) y\f$.  For most common time
/// steppers this is \f$\pi\f$.
void stability_test(const TimeStepper& stepper, double phase = M_PI);

void equal_rate_boundary(const LtsTimeStepper& stepper, size_t order,
                         size_t number_of_past_steps, double epsilon,
                         bool forward);

/// Check that integration converges as expected.
///
/// The \p step_range argument specifies the range of the number of
/// steps used to produce a fit, and should be cover a factor of a few
/// over which a log-log plot of the error is roughly linear.  An
/// appropriate value can be determined by passing `true` as the \p
/// output argument, which will produce a `convergence.dat` file.
void check_convergence_order(const TimeStepper& stepper,
                             const std::pair<int32_t, int32_t>& step_range,
                             bool output = false);

void check_dense_output(const TimeStepper& stepper,
                        const size_t history_integration_order);

void check_boundary_dense_output(const LtsTimeStepper& stepper);

void check_strong_stability_preservation(const TimeStepper& stepper,
                                         double step_size);
}  // namespace TimeStepperTestUtils
