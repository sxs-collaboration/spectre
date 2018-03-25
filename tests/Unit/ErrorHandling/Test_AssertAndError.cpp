// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"

// [[OutputRegex, Testing assert]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ErrorHandling.Assert",
                               "[Unit][ErrorHandling]") {
  ASSERTION_TEST();
  ASSERT(false, "Testing assert");
}

// [[OutputRegex, Testing error]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ErrorHandling.Error",
                               "[Unit][ErrorHandling]") {
  ERROR_TEST();
  ERROR("Testing error");
}
