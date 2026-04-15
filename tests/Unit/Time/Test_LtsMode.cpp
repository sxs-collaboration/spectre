// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Time/LtsMode.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
void test_mode(const LtsMode mode, const std::string& name) {
  CHECK(get_output(mode) == name);
  CHECK(TestHelpers::test_creation<LtsMode>(name) == mode);
}

SPECTRE_TEST_CASE("Unit.Time.LtsMode", "[Unit][Time]") {
#define TEST_LTS_MODE(mode) test_mode(LtsMode::mode, #mode)
  TEST_LTS_MODE(Off);
  TEST_LTS_MODE(Conservative);
#undef TEST_LTS_MODE

  CHECK_THROWS_WITH(TestHelpers::test_creation<LtsMode>("Bad"),
                    Catch::Matchers::ContainsSubstring("Invalid LtsMode"));
}
}  // namespace
