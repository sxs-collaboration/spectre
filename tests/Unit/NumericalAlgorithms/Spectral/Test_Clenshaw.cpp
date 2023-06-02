// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Clenshaw.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral {
namespace {
template <typename BasisFunction, typename DataType>
DataType evaluate_with_function_basis(const std::vector<double>& coefficients,
                                      const DataType& x,
                                      const BasisFunction& f_of_k_and_x) {
  size_t counter = 0;
  return std::accumulate(
      coefficients.begin(), coefficients.end(),
      make_with_value<DataType>(x, 0.0),
      [&x, &f_of_k_and_x, &counter](auto& sum, const auto& coefficient) {
        counter++;
        return sum + coefficient * f_of_k_and_x(counter - 1, x);
      });
}
// Check that for small arguments the clenshaw expansion and the direct cosine
// evaluation agree.

template <typename DataType>
void test_cosine_series(const std::vector<double>& coefficients,
                        const DataType& x) {
  // cos(nx) = 2*x * cos((n-1)x) - cos((n-2)x)
  // cos(nx) = alpha * cos((n-1)x) + beta cos((n-2)x)
  const DataType alpha = 2.0 * cos(x);
  const DataType beta = make_with_value<DataType>(x, -1.0);
  const DataType cos_0x = make_with_value<DataType>(x, 1.0);
  const DataType cos_x = cos(x);
  const DataType true_value = evaluate_with_function_basis(
      coefficients, x, [](const size_t& k, const DataType& z) {
        return cos(static_cast<double>(k) * z);
      });
  const DataType clenshaw_value =
      Spectral::evaluate_clenshaw(coefficients, alpha, beta, cos_0x, cos_x);
  CHECK_ITERABLE_APPROX(clenshaw_value, true_value);
}
// More broad testing allowing a custom approx, in general the clenshaw and
// machine computed series values may not agree to within machine precision.
template <typename DataType>
void test_approximate_cosine_series(const std::vector<double>& coefficients,
                                    const DataType& x, Approx custom_approx) {
  // cos(nx) = 2*cos(x) * cos((n-1)x) - cos((n-2)x)
  // cos(nx) = alpha * cos((n-1)x) + beta cos((n-2)x)
  const DataType alpha = 2.0 * cos(x);
  const DataType beta = make_with_value<DataType>(x, -1.0);
  const DataType cos_0x = make_with_value<DataType>(x, 1.0);
  const DataType cos_x = cos(x);
  const DataType true_value = evaluate_with_function_basis(
      coefficients, x, [](const size_t& k, const DataType& z) {
        return cos(static_cast<double>(k) * z);
      });
  const DataType clenshaw_value =
      Spectral::evaluate_clenshaw(coefficients, alpha, beta, cos_0x, cos_x);
  CHECK_ITERABLE_CUSTOM_APPROX(clenshaw_value, true_value, custom_approx);
}

template <typename DataType>
void test_sine_series(const std::vector<double>& coefficients,
                      const DataType& x) {
  // sin(nx) = 2*cos(x) * sin((n-1)x) - sin((n-2)x)
  // sin(nx) = alpha * sin((n-1)x) + beta sin((n-2)x)
  const DataType alpha = 2.0 * cos(x);
  const DataType beta = make_with_value<DataType>(x, -1.0);
  const DataType sin_0x = make_with_value<DataType>(x, 0.0);
  const DataType sin_x = sin(x);
  const DataType true_value = evaluate_with_function_basis(
      coefficients, x, [](const size_t& k, const DataType& z) {
        return sin(static_cast<double>(k) * z);
      });
  const DataType clenshaw_value =
      Spectral::evaluate_clenshaw(coefficients, alpha, beta, sin_0x, sin_x);
  CHECK_ITERABLE_APPROX(clenshaw_value, true_value);
}

}  // namespace
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Clenshaw",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_cosine_series({}, M_PI);
  test_cosine_series({1.0}, 1.0);
  test_cosine_series({1.0, 2.0, 3.6}, M_PI);
  test_cosine_series({1.0, 2.0, 3.6, -21.1, M_E}, 2.1214353);
  test_cosine_series({0.0, -1.5, 0.0001, -1.1, M_E}, 32.1222);
  test_cosine_series({3200.0, 2100.0, 5065.0, 200000.0, 2211.0}, M_PI * 2.0);

  test_cosine_series(
      {1.0, 2.0, 3.6, 9.5, M_E},
      DataVector{1.0, M_PI, 1.04, 9.0211, 5.1, -92.2, -M_PI / 2.0});

  Approx coarse_approx{1.0e-12};
  CHECK(1.0 == coarse_approx(cos(M_PI * 100000000.0)));
  test_approximate_cosine_series({1.0, 2.0, 3.6}, 200000 * M_PI, coarse_approx);
  test_approximate_cosine_series(
      {1.0, 2.0, 3.6, 9.5, M_E},
      DataVector{100020.0, M_PI * 3000.0, -10002312.0, 1.0e-11, 3.2, -22.1,
                 -M_PI / 2000.0},
      coarse_approx);

  test_sine_series({}, M_PI);
  test_sine_series({1.0}, 1.0);
  test_sine_series({1.0, 2.0, 3.6}, M_PI);
  test_sine_series({1.0, 2.0, 3.6, -21.1, M_E}, 2.1214353);
  test_sine_series({0.0, -1.5, 0.0001, -1.1, M_E}, 32.1222);
  test_sine_series({3200.0, 2100.0, 5065.0, 200000.0, 2211.0}, M_PI * 2.0);

  test_sine_series({1.0, 2.0, 3.6, 9.5, M_E},
                   DataVector{1.0, M_PI, 1.04, 9.0211, 5.1, -9.2, -M_PI / 2.0});
}

}  // namespace Spectral
