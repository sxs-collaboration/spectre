// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>

#include "Options/Options.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order, false);
    TimeStepperTestUtils::check_multistep_properties(stepper);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::integrate_test(stepper, 1., epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order, true);
    TimeStepperTestUtils::check_multistep_properties(stepper);
    // Accuracy limited by first step
    TimeStepperTestUtils::integrate_test(stepper, 1., 1e-3);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Variable",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::integrate_variable_test(
        TimeSteppers::AdamsBashforthN(order, false), epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    // Accuracy limited by first step
    TimeStepperTestUtils::integrate_variable_test(
        TimeSteppers::AdamsBashforthN(order, true), 1e-3);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Backwards",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::integrate_test(
        TimeSteppers::AdamsBashforthN(order, false), -1., epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    // Accuracy limited by first step
    TimeStepperTestUtils::integrate_test(
        TimeSteppers::AdamsBashforthN(order, true), -1., 1e-3);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Stability",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    TimeStepperTestUtils::stability_test(
        TimeSteppers::AdamsBashforthN(order, false));
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Factory",
                  "[Unit][Time]") {
  test_factory_creation<TimeStepper>("  AdamsBashforthN:\n"
                                     "    TargetOrder: 3");
}
