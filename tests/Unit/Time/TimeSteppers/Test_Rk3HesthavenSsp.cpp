// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.Rk3HesthavenSsp", "[Unit][Time]") {
  const TimeSteppers::Rk3HesthavenSsp stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 3, 0, 1., 1e-9);
  TimeStepperTestUtils::integrate_test(stepper, 3, 0, -1., 1e-9);
  TimeStepperTestUtils::integrate_test_explicit_time_dependence(stepper, 3, 0,
                                                                -1.0, 1.0e-9);
  TimeStepperTestUtils::integrate_error_test(stepper, 3, 0, 1.0, 1.0e-8, 100,
                                             1.0e-4);
  TimeStepperTestUtils::integrate_error_test(stepper, 3, 0, -1.0, 1.0e-8, 100,
                                             1.0e-4);
  TimeStepperTestUtils::integrate_variable_test(stepper, 3, 0, 1e-9);
  TimeStepperTestUtils::stability_test(stepper);
  TimeStepperTestUtils::check_convergence_order(stepper);
  TimeStepperTestUtils::check_dense_output(stepper, 3_st);

  CHECK(stepper.order() == 3_st);
  CHECK(stepper.error_estimate_order() == 2_st);

  TestHelpers::test_factory_creation<TimeStepper,
                                     TimeSteppers::Rk3HesthavenSsp>(
      "Rk3HesthavenSsp");
  test_serialization(stepper);
  test_serialization_via_base<TimeStepper, TimeSteppers::Rk3HesthavenSsp>();
  // test operator !=
  CHECK_FALSE(stepper != stepper);
}
