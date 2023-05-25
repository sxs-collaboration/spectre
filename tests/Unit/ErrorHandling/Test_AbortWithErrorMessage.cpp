// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"

SPECTRE_TEST_CASE("Unit.ErrorHandling.AbortWithErrorMessage",
                  "[Unit][ErrorHandling]") {
  CHECK_THROWS_WITH(
      abort_with_error_message("a == b", __FILE__, __LINE__,
                               static_cast<const char*>(__PRETTY_FUNCTION__),
                               "Test Abort"),
      Catch::Contains("ASSERT FAILED") && Catch::Contains("a == b") &&
          Catch::Contains("Test Abort") &&
          Catch::Contains("Test_AbortWithErrorMessage") &&
          Catch::Contains("Stack trace:"));
  CHECK_THROWS_WITH(
      abort_with_error_message(__FILE__, __LINE__,
                               static_cast<const char*>(__PRETTY_FUNCTION__),
                               "Test Error"),
      Catch::Contains("ERROR") && Catch::Contains("Test Error") &&
          Catch::Contains("Test_AbortWithErrorMessage") &&
          Catch::Contains("Stack trace:"));
  CHECK_THROWS_WITH(
      abort_with_error_message_no_trace(
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__),
          "Test no trace"),
      Catch::Contains("ERROR") && Catch::Contains("Test no trace") &&
          Catch::Contains("Test_AbortWithErrorMessage") &&
          not Catch::Contains("Stack trace:"));
  CHECK_THROWS_AS(
      abort_with_error_message_no_trace(
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__),
          "Test no trace"),
      std::runtime_error);
  {
    INFO("Demangling");
    CHECK_THROWS_WITH(
        abort_with_error_message(__FILE__, __LINE__,
                                 static_cast<const char*>(__PRETTY_FUNCTION__),
                                 "Test demangling"),
        Catch::Contains("Catch::Session::run()"));
  }
}
