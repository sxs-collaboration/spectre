// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Time/OptionTags/InitialTimeStep.hpp"

SPECTRE_TEST_CASE("Unit.Time.OptionTags.InitialTimeStep", "[Unit][Time]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::InitialTimeStep>("0.01") ==
        0.01);
}
