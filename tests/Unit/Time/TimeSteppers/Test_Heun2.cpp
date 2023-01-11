// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Heun2", "[Unit][Time]") {
  const TimeSteppers::Heun2 stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 2, 0, 1.0, 1.0e-6);
  TimeStepperTestUtils::integrate_test(stepper, 2, 0, -1.0, 1.0e-6);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 2, 0,
                                                                -1.0, 1.0e-6);
  TimeStepperTestUtils::integrate_error_test(stepper, 2, 0, 1.0, 1.0e-6, 600,
                                             1.0e-4);
  TimeStepperTestUtils::integrate_error_test(stepper, 2, 0, -1.0, 1.0e-6, 600,
                                             1.0e-4);
  TimeStepperTestUtils::integrate_variable_test(stepper, 2, 0, 1.0e-6);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 2_st);

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::Heun2>("Heun2");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Heun2>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
