// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/LinearRegression.hpp"

namespace {

void test_linear_regression_datavector(const double expected_intercept,
                                       const double expected_slope) {
  const DataVector x_values{1.0, 2.0, 3.0, 4.0, 5.0};
  const DataVector y_values = expected_slope * x_values + expected_intercept;
  const auto result = intrp::linear_regression(x_values, y_values);

  CHECK(result.intercept == approx(expected_intercept));
  CHECK(result.slope == approx(expected_slope));
  // Error bars are zero because we have exactly a straight line.
  CHECK(result.delta_intercept == approx(0.0));
  CHECK(result.delta_slope == approx(0.0));
}

void test_linear_regression_stdvectordouble(const double expected_intercept,
                                            const double expected_slope) {
  const std::vector<double> x_values{1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y_values;
  for (size_t i = 0; i < x_values.size(); ++i) {
    y_values.push_back(expected_slope * x_values[i] + expected_intercept);
  }
  const auto result = intrp::linear_regression(x_values, y_values);

  CHECK(result.intercept == approx(expected_intercept));
  CHECK(result.slope == approx(expected_slope));
  // Error bars are zero because we have exactly a straight line.
  CHECK(result.delta_intercept == approx(0.0));
  CHECK(result.delta_slope == approx(0.0));
}

void test_linear_regression_error_bars() {
  const DataVector x_values{1.0, 2.0, 3.0};
  const DataVector y_values{2.0, 4.0, 5.0};

  const auto result = intrp::linear_regression(x_values, y_values);
  // The values below were worked out by hand for the above 3 points.
  CHECK(result.intercept == approx(2.0 / 3.0));
  CHECK(result.slope == approx(3.0 / 2.0));
  CHECK(result.delta_intercept == approx(sqrt(7.0 / 18.0)));
  CHECK(result.delta_slope == approx(sqrt(1.0 / 12.0)));
}

void test_linear_regression() {
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> coeff_dis(-10.0, 10.0);

  const double intercept = coeff_dis(gen);
  CAPTURE(intercept);
  const double slope = coeff_dis(gen);
  CAPTURE(slope);

  test_linear_regression_stdvectordouble(intercept, slope);
  test_linear_regression_datavector(intercept, slope);
  test_linear_regression_error_bars();
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.LinearRegression",
                  "[Unit][NumericalAlgorithms]") {
  test_linear_regression();
}
