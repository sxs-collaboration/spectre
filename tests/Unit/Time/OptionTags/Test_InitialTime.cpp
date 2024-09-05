// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Time/OptionTags/InitialTime.hpp"

SPECTRE_TEST_CASE("Unit.Time.OptionTags.InitialTime", "[Unit][Time]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::InitialTime>("0.01") == 0.01);
}
