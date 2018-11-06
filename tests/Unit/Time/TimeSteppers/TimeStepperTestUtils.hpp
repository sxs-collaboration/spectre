// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "Time/TimeSteppers/TimeStepper.hpp"

namespace TimeStepperTestUtils {

inline void check_multistep_properties(const TimeStepper& stepper) noexcept {
  CHECK(stepper.number_of_substeps() == 1);
}

inline void check_substep_properties(const TimeStepper& stepper) noexcept {
  CHECK(stepper.number_of_past_steps() == 0);
}

void integrate_test(const TimeStepper& stepper, size_t number_of_past_steps,
                    double integration_time, double epsilon) noexcept;

void integrate_variable_test(const TimeStepper& stepper,
                             size_t number_of_past_steps,
                             double epsilon) noexcept;

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
