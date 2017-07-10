// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3", "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  check_substep_properties(stepper);
  integrate_test(stepper, 1., 1e-9);
}

TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Variable", "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  integrate_variable_test(stepper, 1e-9);
}

TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Backwards", "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  integrate_test(stepper, -1., 1e-9);
}

TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Stability", "[Unit][Time]") {
  stability_test(TimeSteppers::RungeKutta3{});
}
