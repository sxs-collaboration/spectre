// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "tests/Unit/TestFactoryCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3", "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::check_substep_properties(stepper);
  TimeStepperTestUtils::integrate_test(stepper, 1., 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Variable",
                  "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::integrate_variable_test(stepper, 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Backwards",
                  "[Unit][Time]") {
  const TimeSteppers::RungeKutta3 stepper{};
  TimeStepperTestUtils::integrate_test(stepper, -1., 1e-9);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Stability",
                  "[Unit][Time]") {
  TimeStepperTestUtils::stability_test(TimeSteppers::RungeKutta3{});
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Factory",
                  "[Unit][Time]") {
  test_factory_creation<TimeStepper>("  RungeKutta3");
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Boundary.Equal",
                  "[Unit][Time]") {
  TimeStepperTestUtils::equal_rate_boundary(
      TimeSteppers::RungeKutta3{}, 1e-9, true);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Boundary.Equal.Backwards",
                  "[Unit][Time]") {
  TimeStepperTestUtils::equal_rate_boundary(
      TimeSteppers::RungeKutta3{}, 1e-9, false);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.RungeKutta3.Serialization",
                  "[Unit][Time]") {
  register_derived_classes_with_charm<TimeStepper>();
  std::unique_ptr<TimeStepper> stepper =
      std::make_unique<TimeSteppers::RungeKutta3>();
  std::unique_ptr<TimeStepper> stepper_puped =
      serialize_and_deserialize(stepper);
  auto stepper_cast =
      dynamic_cast<TimeSteppers::RungeKutta3* const>(stepper_puped.get());
  CHECK_FALSE(stepper_cast == nullptr);
}
