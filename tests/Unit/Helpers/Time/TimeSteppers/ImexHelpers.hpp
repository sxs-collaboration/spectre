// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <utility>

/// \cond
class ImexTimeStepper;
/// \endcond

namespace TimeStepperTestUtils::imex {
void check_convergence_order(const ImexTimeStepper& stepper,
                             const std::pair<int32_t, int32_t>& step_range,
                             bool output = false);
}  // namespace TimeStepperTestUtils::imex
