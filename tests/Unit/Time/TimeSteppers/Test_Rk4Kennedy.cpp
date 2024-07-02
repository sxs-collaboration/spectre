// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/ImexHelpers.hpp"
#include "Helpers/Time/TimeSteppers/RungeKutta.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Rk4Kennedy.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Rk4Kennedy", "[Unit][Time]") {
  const TimeSteppers::Rk4Kennedy stepper{};

  CHECK(stepper.order() == 4);
  CHECK(stepper.number_of_substeps() == 6);
  CHECK(stepper.number_of_substeps_for_error() == 6);
  TestHelpers::RungeKutta::check_tableau(stepper);
  TestHelpers::RungeKutta::check_implicit_tableau(stepper, true);

  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 4, 0, 1.0, 1.0e-14);
  TimeStepperTestUtils::integrate_test(stepper, 4, 0, -1.0, 1.0e-14);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 4, 0,
                                                                -1.0, 1.0e-14);
  TimeStepperTestUtils::integrate_error_test(stepper, 4, 0, 1.0, 1.0e-8, 50,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_error_test(stepper, 4, 0, -1.0, 1.0e-8, 50,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_variable_test(stepper, 4, 0, 1.0e-14);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper, {10, 50});
  TimeStepperTestUtils::check_dense_output(stepper, {10, 30}, 1, true);

  TimeStepperTestUtils::imex::check_convergence_order(stepper, {10, 50});
  TimeStepperTestUtils::imex::check_bounded_dense_output(stepper);

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::Rk4Kennedy>(
      "Rk4Kennedy");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Rk4Kennedy>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
