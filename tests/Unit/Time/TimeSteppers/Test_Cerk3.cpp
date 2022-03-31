// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Cerk3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Cerk3", "[Unit][Time]") {
  const TimeSteppers::Cerk3 stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 3, 0, 1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test(stepper, 3, 0, -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 3, 0,
                                                                -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_error_test(stepper, 3, 0, 1.0, 1.0e-6, 100,
                                             1.0e-3);
  TimeStepperTestUtils::integrate_error_test(stepper, 3, 0, -1.0, 1.0e-6, 100,
                                             1.0e-3);
  TimeStepperTestUtils::integrate_variable_test(stepper, 3, 0, 1.0e-9);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 3_st);

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::Cerk3>("Cerk3");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Cerk3>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
