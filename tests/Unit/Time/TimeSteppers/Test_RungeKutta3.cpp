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
  TimeStepperTestUtils::integrate_test(stepper, 0, -1., 1e-9);
  TimeStepperTestUtils::integrate_variable_test(stepper, 0, 1e-9);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper, 3);
  TimeStepperTestUtils::check_dense_output(stepper, 3);

  test_factory_creation<TimeStepper>("  RungeKutta3");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::RungeKutta3>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
