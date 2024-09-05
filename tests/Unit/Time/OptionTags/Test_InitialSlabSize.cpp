// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Time/OptionTags/InitialSlabSize.hpp"

SPECTRE_TEST_CASE("Unit.Time.OptionTags.InitialSlabSize", "[Unit][Time]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::InitialSlabSize>("0.01") ==
        0.01);
}
