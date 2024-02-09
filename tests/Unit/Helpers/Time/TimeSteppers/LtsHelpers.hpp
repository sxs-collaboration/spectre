// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class LtsTimeStepper;
/// \endcond

namespace TimeStepperTestUtils::lts {
/// Test boundary computations with the same step size on both
/// neighbors.
void test_equal_rate(const LtsTimeStepper& stepper, size_t order,
                     size_t number_of_past_steps, double epsilon, bool forward);


/// Test the accuracy of boundary dense output.
void test_dense_output(const LtsTimeStepper& stepper);
}  // namespace TimeStepperTestUtils::lts
