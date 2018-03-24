// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "ErrorHandling/AbortWithErrorMessage.hpp"

/// [error_test_example]
// [[OutputRegex, 'a == b' violated!]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.ErrorHandling.AbortWithErrorMessage.Assert",
    "[Unit][ErrorHandling]") {
  ERROR_TEST();
  abort_with_error_message("a == b", __FILE__, __LINE__,
                           static_cast<const char*>(__PRETTY_FUNCTION__),
                           "Test Error");
}
/// [error_test_example]

// [[OutputRegex, ############ ERROR]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ErrorHandling.AbortWithErrorMessage.Error",
                               "[Unit][ErrorHandling]") {
  ERROR_TEST();
  abort_with_error_message(__FILE__, __LINE__,
                           static_cast<const char*>(__PRETTY_FUNCTION__),
                           "Test Error");
}
