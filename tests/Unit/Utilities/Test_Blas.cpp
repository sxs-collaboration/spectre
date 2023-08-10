// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/Test_Blas.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Blas", "[Unit][Utilities]") {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      test_blas_asserts_for_bad_char::dgemm_error_transa_false(),
      Catch::Matchers::ContainsSubstring(
          "TRANSA must be upper or lower case N, T, or C. See the "
          "BLAS documentation for help."));
  CHECK_THROWS_WITH(
      test_blas_asserts_for_bad_char::dgemm_error_transb_false(),
      Catch::Matchers::ContainsSubstring(
          "TRANSB must be upper or lower case N, T, or C. See the "
          "BLAS documentation for help."));
  CHECK_THROWS_WITH(
      test_blas_asserts_for_bad_char::dgemm_error_transa_true(),
      Catch::Matchers::ContainsSubstring(
          "TRANSA must be upper or lower case N, T, or C. See the "
          "BLAS documentation for help."));
  CHECK_THROWS_WITH(
      test_blas_asserts_for_bad_char::dgemm_error_transb_true(),
      Catch::Matchers::ContainsSubstring(
          "TRANSB must be upper or lower case N, T, or C. See the "
          "BLAS documentation for help."));
  CHECK_THROWS_WITH(test_blas_asserts_for_bad_char::dgemv_error_trans(),
                    Catch::Matchers::ContainsSubstring(
                        "TRANS must be upper or lower case N, T, or C. See the "
                        "BLAS documentation for help."));
#endif
  // so test does not fail in release mode
  CHECK(true);
}
