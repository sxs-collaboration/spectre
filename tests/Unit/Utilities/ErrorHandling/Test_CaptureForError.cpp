// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ErrorHandling.CaptureForError",
                  "[Unit][ErrorHandling]") {
  CHECK_THROWS_WITH(
      []() { ERROR("Boom"); }(),
      not Catch::Matchers::ContainsSubstring("Captured variables:"));

  std::vector<int> vector_var{1, 2, 3};
  const int int_var = 7;
  CAPTURE_FOR_ERROR(vector_var);
  CAPTURE_FOR_ERROR(int_var);
  vector_var.push_back(4);
  CHECK_THROWS_WITH(
      []() { ERROR("Boom"); }(),
      Catch::Matchers::ContainsSubstring("Captured variables:\n"
                                         "vector_var = (1,2,3,4)\n"
                                         "int_var = 7"));
  // Check that the list is left in the same state.
  CHECK_THROWS_WITH(
      []() { ERROR("Boom"); }(),
      Catch::Matchers::ContainsSubstring("Captured variables:\n"
                                         "vector_var = (1,2,3,4)\n"
                                         "int_var = 7"));
  // Test an additional capture in a nested scope
  CHECK_THROWS_WITH(
      []() {
        const int another_int = 12;
        CAPTURE_FOR_ERROR(another_int);
        ERROR("Boom");
      }(),
      Catch::Matchers::ContainsSubstring("Captured variables:\n"
                                         "vector_var = (1,2,3,4)\n"
                                         "int_var = 7\n"
                                         "another_int = 12"));
  // And make sure it's gone again.
  CHECK_THROWS_WITH([]() { ERROR("Boom"); }(),
                    not Catch::Matchers::ContainsSubstring("another_int"));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      []() { ASSERT(false, "Boom"); }(),
      Catch::Matchers::ContainsSubstring("Captured variables:\n"
                                         "vector_var = (1,2,3,4)\n"
                                         "int_var = 7"));
#endif  // SPECTRE_DEBUG
}
