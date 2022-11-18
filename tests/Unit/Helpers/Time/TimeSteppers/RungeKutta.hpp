// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/TimeSteppers/RungeKutta.hpp"

namespace TestHelpers::RungeKutta {
/// Sanity-check a Butcher tableau
void check_tableau(const TimeSteppers::RungeKutta::ButcherTableau& tableau,
                   size_t expected_order, size_t expected_error_order);
/// Convenience wrapper for the previous function
void check_tableau(const TimeSteppers::RungeKutta& stepper);
}  // namespace TestHelpers::RungeKutta
