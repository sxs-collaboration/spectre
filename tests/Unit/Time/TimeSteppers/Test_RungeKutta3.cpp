// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Parallel/PupStlCpp11.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3", "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 0, 1., 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Variable",
                  "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::integrate_variable_test(stepper, 0, 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Backwards",
                  "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::integrate_test(stepper, 0, -1., 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Stability",
                  "[Unit][Time]") {
  TimeStepperTestUtils::stability_test(TimeSteppers::RungeKutta3{});
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Factory",
                  "[Unit][Time]") {
  test_factory_creation<TimeStepper>("  RungeKutta3");
  // Catch requires us to have at least one CHECK in each test
  // The Unit.Time.TimeSteppers.RungeKutta3.Factory does not need to
  // check anything
  CHECK(true);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Serialization",
                  "[Unit][Time]") {
  TimeSteppers::RungeKutta3 rk3{};
  test_serialization(rk3);
  test_serialization_via_base<TimeStepper, TimeSteppers::RungeKutta3>();
  // test operator !=
  CHECK_FALSE(rk3 != rk3);
}
