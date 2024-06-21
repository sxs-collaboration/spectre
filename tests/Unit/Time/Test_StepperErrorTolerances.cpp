// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestHelpers.hpp"
#include "Time/StepperErrorTolerances.hpp"

SPECTRE_TEST_CASE("Unit.Time.StepperErrorTolerances", "[Unit][Time]") {
  const StepperErrorTolerances tols1{.absolute = 0.1, .relative = 0.3};
  const StepperErrorTolerances tols2{.absolute = 0.1, .relative = 0.5};
  const StepperErrorTolerances tols3{.absolute = 0.5, .relative = 0.3};
  CHECK(tols1 == tols1);
  CHECK_FALSE(tols1 != tols1);
  CHECK(tols1 != tols2);
  CHECK_FALSE(tols1 == tols2);
  CHECK(tols1 != tols3);
  CHECK_FALSE(tols1 == tols3);
  test_serialization(tols1);
}
