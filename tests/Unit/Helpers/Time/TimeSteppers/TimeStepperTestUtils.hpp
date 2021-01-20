// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class LtsTimeStepper;
class TimeStepper;
/// \endcond

namespace TimeStepperTestUtils {

void check_multistep_properties(const TimeStepper& stepper) noexcept;

void check_substep_properties(const TimeStepper& stepper) noexcept;

void integrate_test(const TimeStepper& stepper, size_t number_of_past_steps,
                    double integration_time, double epsilon) noexcept;

void integrate_test_explicit_time_dependence(const TimeStepper& stepper,
                                             size_t number_of_past_steps,
                                             double integration_time,
                                             double epsilon) noexcept;

void integrate_variable_test(const TimeStepper& stepper,
                             size_t number_of_past_steps,
                             double epsilon) noexcept;

void integrate_error_test(const TimeStepper& stepper,
                          size_t number_of_past_steps, double integration_time,
                          double epsilon, size_t num_steps,
                          double error_factor) noexcept;

void stability_test(const TimeStepper& stepper) noexcept;

void equal_rate_boundary(const LtsTimeStepper& stepper,
                         size_t number_of_past_steps,
                         double epsilon, bool forward) noexcept;

void check_convergence_order(const TimeStepper& stepper,
                             int expected_order) noexcept;

void check_dense_output(const TimeStepper& stepper,
                        int expected_order) noexcept;

void check_boundary_dense_output(const LtsTimeStepper& stepper) noexcept;
}  // namespace TimeStepperTestUtils
