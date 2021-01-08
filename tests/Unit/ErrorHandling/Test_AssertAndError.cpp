// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

/// [assertion_test_example]
// [[OutputRegex, Testing assert]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ErrorHandling.Assert",
                               "[Unit][ErrorHandling]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  ASSERT(false, "Testing assert");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
/// [assertion_test_example]

/// [error_test_example]
// [[OutputRegex, Testing error]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ErrorHandling.Error",
                               "[Unit][ErrorHandling]") {
  ERROR_TEST();
  ERROR("Testing error");
}
/// [error_test_example]
