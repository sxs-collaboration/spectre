// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Time/LtsMode.hpp"
#include "Time/OptionTags/LocalTimeStepping.hpp"

SPECTRE_TEST_CASE("Unit.Time.OptionTags.LocalTimeStepping", "[Unit][Time]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::LocalTimeStepping>(
            "Conservative") == LtsMode::Conservative);
}
