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
  // Verify that zero-sized matrix multiplications are treated as no-ops.
  const size_t zero = 0;
  const size_t one = 1;
  const double alpha = 1.23;
  const double beta = 4.56;
  const std::array<double, 1> a{{7.89}};
  const std::array<double, 1> b{{0.12}};
  std::array<double, 1> c{{3.45}};
  const double original_c = c[0];
  dgemm_('N', 'N', zero, zero, zero, alpha, a.data(), zero, b.data(), one, beta,
         c.data(), one);
  CHECK(c[0] == original_c);

  dgemv_('N', zero, zero, alpha, a.data(), zero, b.data(), one, beta, c.data(),
         one);
  CHECK(c[0] == original_c);

  const std::complex<double> alpha_complex = std::complex<double>{1.23, 3.21};
  const std::complex<double> beta_complex = std::complex<double>{4.56, 6.54};
  const std::array<std::complex<double>, 1> a_complex{
      {std::complex<double>{7.89, 9.87}}};
  const std::array<std::complex<double>, 1> b_complex{
      {std::complex<double>{0.12, 2.10}}};
  std::array<std::complex<double>, 1> c_complex{
      {std::complex<double>{3.45, 5.43}}};
  const std::complex<double> original_c_complex = c_complex[0];
  zgemm_('N', 'N', zero, zero, zero, alpha_complex, a_complex.data(), zero,
         b_complex.data(), one, beta_complex, c_complex.data(), one);
  CHECK(c_complex[0] == original_c_complex);
}
