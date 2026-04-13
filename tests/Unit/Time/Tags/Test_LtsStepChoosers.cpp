// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <pup_stl.h>
#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/LtsStepChoosers.hpp"

SPECTRE_TEST_CASE("Unit.Time.Tags.LtsStepChoosers", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::LtsStepChoosers>("LtsStepChoosers");
}
