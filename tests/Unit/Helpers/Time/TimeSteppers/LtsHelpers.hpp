// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

/// \cond
class LtsTimeStepper;
/// \endcond

namespace TimeStepperTestUtils::lts {
struct VariableOrderChoice {
  size_t local_order;
  size_t local_number_of_past_steps;
  size_t remote_order;
  size_t remote_number_of_past_steps;
};

/// Test boundary computations with the same step size on both
/// neighbors against the volume computation.
void test_equal_rate(const LtsTimeStepper& stepper);

/// Test uncoupled boundary computations against the volume
/// computation.
void test_uncoupled(
    const LtsTimeStepper& stepper, double tolerance,
    std::optional<VariableOrderChoice> variable_order_choice = std::nullopt);

/// Test conservation of boundary dense output.
void test_conservation(
    const LtsTimeStepper& stepper,
    std::optional<VariableOrderChoice> variable_order_choice = std::nullopt);

// Test convergence rate of boundary integration.
void test_convergence(
    const LtsTimeStepper& stepper,
    const std::pair<int32_t, int32_t>& number_of_steps_range, int32_t stride,
    std::optional<VariableOrderChoice> variable_order_choice = std::nullopt);

// Test convergence rate of boundary dense output.
void test_dense_convergence(
    const LtsTimeStepper& stepper,
    const std::pair<int32_t, int32_t>& number_of_steps_range, int32_t stride,
    std::optional<VariableOrderChoice> variable_order_choice = std::nullopt);
}  // namespace TimeStepperTestUtils::lts
