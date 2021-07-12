// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/RungeKutta4.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta4", "[Unit][Time]") {
  const TimeSteppers::RungeKutta4 stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 4, 0, 1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test(stepper, 4, 0, -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 4, 0,
                                                                -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_error_test(stepper, 4, 0, 1.0, 1.0e-8, 50,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_error_test(stepper, 4, 0, -1.0, 1.0e-8, 50,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_variable_test(stepper, 4, 0, 1.0e-9);
  TimeStepperTestUtils::check_convergence_order(stepper);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 4_st);

  CHECK(stepper.order() == 4_st);
  CHECK(stepper.error_estimate_order() == 3_st);

  TestHelpers::test_creation<std::unique_ptr<TimeStepper>>("RungeKutta4");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::RungeKutta4>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
