// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace {
#ifdef SPECTRE_DEBUG
[[noreturn]] void trigger_assert() { ASSERT(false, "Testing assert"); }
#endif

[[noreturn]] void trigger_error() { ERROR("Testing error"); }
}  // namespace

SPECTRE_TEST_CASE("Unit.ErrorHandling.AssertAndError",
                  "[Unit][ErrorHandling]") {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(trigger_assert(), Catch::Contains("Testing assert") &&
                                          Catch::Contains("false"));
#endif
  CHECK_THROWS_WITH(trigger_error(), Catch::Contains("Testing error"));
}
