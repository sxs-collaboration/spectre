// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "ErrorHandling/AbortWithErrorMessage.hpp"

// [[OutputRegex, 'a == b' violated!]]
[[noreturn]] TEST_CASE("Unit.ErrorHandling.AbortWithErrorMessage.Assert",
                       "[Unit][ErrorHandling]") {
  abort_with_error_message("a == b", __FILE__, __LINE__, "Test Error");
}

// [[OutputRegex, ############ ERROR]]
[[noreturn]] TEST_CASE("Unit.ErrorHandling.AbortWithErrorMessage.Error",
                       "[Unit][ErrorHandling]") {
  abort_with_error_message(__FILE__, __LINE__, "Test Error");
}
