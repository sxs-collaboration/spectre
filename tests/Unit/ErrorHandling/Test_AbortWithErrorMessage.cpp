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
      Catch::Matchers::ContainsSubstring("ASSERT FAILED") &&
          Catch::Matchers::ContainsSubstring("a == b") &&
          Catch::Matchers::ContainsSubstring("Test Abort") &&
          Catch::Matchers::ContainsSubstring("Test_AbortWithErrorMessage") &&
          Catch::Matchers::ContainsSubstring("Stack trace:"));
  CHECK_THROWS_WITH(
      abort_with_error_message<SpectreError>(
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__),
          "Test Error"),
      Catch::Matchers::ContainsSubstring("ERROR") &&
          Catch::Matchers::ContainsSubstring("Test Error") &&
          Catch::Matchers::ContainsSubstring("Test_AbortWithErrorMessage") &&
          Catch::Matchers::ContainsSubstring("Stack trace:"));
  CHECK_THROWS_WITH(
      abort_with_error_message_no_trace(
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__),
          "Test no trace"),
      Catch::Matchers::ContainsSubstring("ERROR") &&
          Catch::Matchers::ContainsSubstring("Test no trace") &&
          Catch::Matchers::ContainsSubstring("Test_AbortWithErrorMessage") &&
          not Catch::Matchers::ContainsSubstring("Stack trace:"));
  CHECK_THROWS_AS(
      abort_with_error_message_no_trace(
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__),
          "Test no trace"),
      std::runtime_error);
  {
    INFO("Demangling");
    CHECK_THROWS_WITH(
        abort_with_error_message<SpectreError>(
            __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__),
            "Test demangling"),
        Catch::Matchers::ContainsSubstring("Catch::Session::run()"));
  }
}
