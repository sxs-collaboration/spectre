// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "tests/Unit/TestFactoryCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3", "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 1., 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Variable",
                  "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::integrate_variable_test(stepper, 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Backwards",
                  "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::integrate_test(stepper, -1., 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Stability",
                  "[Unit][Time]") {
  TimeStepperTestUtils::stability_test(TimeSteppers::RungeKutta3{});
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Factory",
                  "[Unit][Time]") {
  test_factory_creation<TimeStepper>("  RungeKutta3");
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Boundary.Equal",
                  "[Unit][Time]") {
  TimeStepperTestUtils::equal_rate_boundary(
      TimeSteppers::RungeKutta3{}, 1e-9, true);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Boundary.Equal.Backwards",
                  "[Unit][Time]") {
  TimeStepperTestUtils::equal_rate_boundary(
      TimeSteppers::RungeKutta3{}, 1e-9, false);
}
