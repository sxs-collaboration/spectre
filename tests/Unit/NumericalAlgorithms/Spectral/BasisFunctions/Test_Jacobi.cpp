// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Jacobi.hpp"

namespace Spectral {
namespace {
double binomial(const double alpha, const size_t k) {
  double result = 1.0;
  for (uint64_t j = 1; j <= k; ++j) {
    const auto jj = static_cast<double>(j);
    result *= alpha + 1.0 - jj;
    result /= jj;
  }
  return result;
}

void test_values() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> x_distribution(0.0, 1.0);
  const double x = x_distribution(generator);
  CAPTURE(x);
  for (size_t k = 0; k < 20; ++k) {
    CHECK(Jacobi::basis_function_value(0.0, 0.0, k, x) ==
          approx(compute_basis_function_value<Basis::Legendre>(k, x)));
    CHECK(Jacobi::basis_function_value(-0.5, -0.5, k, x) /
              Jacobi::basis_function_value(-0.5, -0.5, k, 1.0) ==
          approx(compute_basis_function_value<Basis::Chebyshev>(k, x)));
  }
  std::uniform_real_distribution<> greek_distribution(-1.0, 10.0);
  const double alpha = greek_distribution(generator);
  const double beta = greek_distribution(generator);
  CAPTURE(alpha);
  CAPTURE(beta);
  for (size_t n = 0; n < 10; ++n) {
    CAPTURE(n);
    const auto nn = static_cast<double>(n);
    double expected_value = 0.0;
    const double scale = pow(0.5, nn);
    for (size_t k = 0; k <= n; ++k) {
      const auto kk = static_cast<double>(k);
      expected_value += scale * binomial(nn + alpha, k) *
                        binomial(nn + beta, n - k) * pow(x - 1.0, nn - kk) *
                        pow(x + 1.0, kk);
    }
    // the binomial formula is not an accurate way to compute the values...
    const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
    CHECK(Jacobi::basis_function_value(alpha, beta, n, x) ==
          custom_approx(expected_value));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctions.Jacobi",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_values();
}
}  // namespace Spectral
