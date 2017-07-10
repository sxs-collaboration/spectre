// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>

#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order, false);
    check_multistep_properties(stepper);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    integrate_test(stepper, 1., epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order, true);
    check_multistep_properties(stepper);
    // Accuracy limited by first step
    integrate_test(stepper, 1., 1e-3);
  }
}

TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Variable", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    integrate_variable_test(TimeSteppers::AdamsBashforthN(order, false),
                            epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    // Accuracy limited by first step
    integrate_variable_test(TimeSteppers::AdamsBashforthN(order, true), 1e-3);
  }
}

TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Backwards", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    integrate_test(TimeSteppers::AdamsBashforthN(order, false), -1., epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    // Accuracy limited by first step
    integrate_test(TimeSteppers::AdamsBashforthN(order, true), -1., 1e-3);
  }
}

TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Stability", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    stability_test(TimeSteppers::AdamsBashforthN(order, false));
  }
}
