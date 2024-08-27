// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/TimeSteppers/ImexRungeKutta.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"

namespace TestHelpers::RungeKutta {
/// Sanity-check a Butcher tableau
void check_tableau(const TimeSteppers::RungeKutta::ButcherTableau& tableau,
                   size_t expected_order);
/// Convenience wrapper for the previous function
void check_tableau(const TimeSteppers::RungeKutta& stepper);

/// Sanity-check an implicit Butcher tableau
void check_implicit_tableau(
    const TimeSteppers::RungeKutta::ButcherTableau& explicit_tableau,
    const TimeSteppers::ImexRungeKutta::ImplicitButcherTableau&
        implicit_tableau,
    size_t expected_stage_order, bool stiffly_accurate);
/// Convenience wrapper for the previous function
void check_implicit_tableau(const TimeSteppers::ImexRungeKutta& stepper,
                            bool stiffly_accurate);
}  // namespace TestHelpers::RungeKutta
