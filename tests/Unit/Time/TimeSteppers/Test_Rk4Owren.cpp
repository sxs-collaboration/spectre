// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Rk4Owren.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Rk4Owren", "[Unit][Time]") {
  const TimeSteppers::Rk4Owren stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 4, 0, 1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test(stepper, 4, 0, -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 4, 0,
                                                                -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_error_test(stepper, 4, 0, 1.0, 1.0e-8, 20,
                                             1.0e-3);
  TimeStepperTestUtils::integrate_error_test(stepper, 4, 0, -1.0, 1.0e-8, 20,
                                             1.0e-3);
  TimeStepperTestUtils::integrate_variable_test(stepper, 4, 0, 1.0e-9);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 4_st);

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::Rk4Owren>(
      "Rk4Owren");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Rk4Owren>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
