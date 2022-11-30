// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Rk5Owren.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Rk5Owren", "[Unit][Time]") {
  const TimeSteppers::Rk5Owren stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 5, 0, 1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test(stepper, 5, 0, -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 5, 0,
                                                                -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_error_test(stepper, 5, 0, 1.0, 1.0e-8, 10,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_error_test(stepper, 5, 0, -1.0, 1.0e-8, 10,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_variable_test(stepper, 5, 0, 1.0e-9);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 5_st);

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::Rk5Owren>(
      "Rk5Owren");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Rk5Owren>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
