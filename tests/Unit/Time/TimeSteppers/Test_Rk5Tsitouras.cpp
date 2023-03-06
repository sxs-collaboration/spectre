// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/RungeKutta.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Rk5Tsitouras.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Rk5Tsitouras", "[Unit][Time]") {
  const TimeSteppers::Rk5Tsitouras stepper{};

  CHECK(stepper.order() == 5);
  CHECK(stepper.error_estimate_order() == 4);
  CHECK(stepper.number_of_substeps() == 6);
  CHECK(stepper.number_of_substeps_for_error() == 7);
  TestHelpers::RungeKutta::check_tableau(stepper);

  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 5, 0, 1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test(stepper, 5, 0, -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 5, 0,
                                                                -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_error_test(stepper, 5, 0, 1.0, 1.0e-8, 8,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_error_test(stepper, 5, 0, -1.0, 1.0e-8, 8,
                                             1.0e-2);
  TimeStepperTestUtils::integrate_variable_test(stepper, 5, 0, 1.0e-9);
  TimeStepperTestUtils::check_convergence_order(stepper, {30, 70});
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 5_st);

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::Rk5Tsitouras>(
      "Rk5Tsitouras");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Rk5Tsitouras>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
