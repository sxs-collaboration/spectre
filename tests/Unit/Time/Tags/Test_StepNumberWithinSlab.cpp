// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"

SPECTRE_TEST_CASE("Unit.Time.Tags.StepNumberWithinSlab", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::StepNumberWithinSlab>(
      "StepNumberWithinSlab");
}
