// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestHelpers.hpp"
#include "Time/StepperErrorTolerances.hpp"

SPECTRE_TEST_CASE("Unit.Time.StepperErrorTolerances", "[Unit][Time]") {
  const StepperErrorTolerances tols0{};
  const StepperErrorTolerances tols1{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 0.1,
      .relative = 0.3};
  const StepperErrorTolerances tols2{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 0.1,
      .relative = 0.5};
  const StepperErrorTolerances tols3{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = 0.5,
      .relative = 0.3};
  const StepperErrorTolerances tols4{
      .estimates = StepperErrorTolerances::Estimates::AllOrders,
      .absolute = 0.1,
      .relative = 0.3};
  CHECK(tols0 == tols0);
  CHECK_FALSE(tols0 != tols0);
  CHECK(tols1 == tols1);
  CHECK_FALSE(tols1 != tols1);
  CHECK(tols1 != tols0);
  CHECK_FALSE(tols1 == tols0);
  CHECK(tols1 != tols2);
  CHECK_FALSE(tols1 == tols2);
  CHECK(tols1 != tols3);
  CHECK_FALSE(tols1 == tols3);
  CHECK(tols1 != tols4);
  CHECK_FALSE(tols1 == tols4);
  test_serialization(tols1);
}
