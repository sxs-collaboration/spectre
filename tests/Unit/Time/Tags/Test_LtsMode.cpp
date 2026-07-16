// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/LtsMode.hpp"

SPECTRE_TEST_CASE("Unit.Time.Tags.LtsMode", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::LtsMode>("LtsMode");
  TestHelpers::db::test_simple_tag<Tags::LtsModeForced<LtsMode::Off>>(
      "LtsMode");
}
