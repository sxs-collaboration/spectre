// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

namespace {
template <size_t Order>
void test_linear_least_squares_double(
    const std::array<double, Order + 1>& coeffs,
    const std::vector<double>& x_values) {
  std::vector<double> y_values{};
  for (size_t i = 0; i < x_values.size(); i++) {
    double y_i = 0;
    for (size_t j = 0; j < Order + 1; j++) {
      y_i += gsl::at(coeffs, j) * pow(gsl::at(x_values, i), j);
    }
    y_values.push_back(y_i);
  }

  // Check that the coeffs determined match the ones used to
  // produce the data points.
  intrp::LinearLeastSquares<Order> lls(x_values.size());
  Approx my_approx = Approx::custom().epsilon(1.e-11).scale(1.0);
  auto deserialized_lls = serialize_and_deserialize(lls);
  const std::array<double, Order + 1> computed_coeffs =
      lls.fit_coefficients(x_values, y_values);
  CHECK_ITERABLE_CUSTOM_APPROX(coeffs, computed_coeffs, my_approx);
  for (size_t i = 0; i < x_values.size(); i++) {
    CHECK(lls.interpolate(computed_coeffs, gsl::at(x_values, i)) ==
          my_approx(gsl::at(y_values, i)));
    CHECK(deserialized_lls.interpolate(computed_coeffs, gsl::at(x_values, i)) ==
          my_approx(gsl::at(y_values, i)));
  }
}

void test_linear_least_squares_datavector() {
  const DataVector x_values{1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<std::array<double, 2>> coefficients{
      {{6.0, 7.0}}, {{8.0, 9.0}}, {{10.0, 11.0}}, {{12.0, 13.0}}};
  std::vector<DataVector> y_values{4, DataVector{5}};
  for (size_t i = 0; i < y_values.size(); ++i) {
    y_values[i] = x_values * coefficients[i][1] + coefficients[i][0];
  }
  intrp::LinearLeastSquares<1> lls(x_values.size());
  const std::vector<std::array<double, 2>> computed_coefficients =
      lls.fit_coefficients(x_values, y_values);
  CHECK_ITERABLE_APPROX(coefficients, computed_coefficients);
  const double x_rand = 2.31;
  for (size_t i = 0; i < y_values.size(); ++i) {
    double expected_y_value = x_rand * coefficients[i][1] + coefficients[i][0];
    double computed_y_value = lls.interpolate(coefficients[i], x_rand);
    CHECK(expected_y_value == computed_y_value);
  }
}

template <size_t Order>
void test_linear_least_squares_datavector2(
    const std::vector<std::array<double, Order + 1>>& coeffs,
    const DataVector& x_values) {
  std::vector<DataVector> y_values{coeffs.size(),
                                   DataVector{x_values.size(), 0.0}};
  for (size_t i = 0; i < y_values.size(); ++i) {
    y_values[i] = evaluate_polynomial(coeffs[i], x_values);
  }
  intrp::LinearLeastSquares<Order> lls(x_values.size());
  const std::vector<std::array<double, Order + 1>> computed_coefficients =
      lls.fit_coefficients(x_values, y_values);
  Approx my_approx = Approx::custom().epsilon(1.e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(coeffs, computed_coefficients, my_approx);
  const double x_rand = 4.21;
  for (size_t i = 0; i < y_values.size(); ++i) {
    double expected_y_value = 0.0;
    for (size_t j = 0; j < Order + 1; j++) {
      expected_y_value += pow(x_rand, j) * coeffs[i][j];
    }
    double computed_y_value = lls.interpolate(coeffs[i], x_rand);
    CHECK(expected_y_value == computed_y_value);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.LinearLeastSquares",
                  "[Unit][NumericalAlgorithms]") {
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> coeff_dis(-10.0, 10.0);
  const double coeff0 = coeff_dis(gen);
  CAPTURE(coeff0);
  const double coeff1 = coeff_dis(gen);
  CAPTURE(coeff1);
  const double coeff2 = coeff_dis(gen);
  CAPTURE(coeff2);
  const double coeff3 = coeff_dis(gen);
  CAPTURE(coeff3);
  const double coeff4 = coeff_dis(gen);
  CAPTURE(coeff4);
  const double coeff5 = coeff_dis(gen);
  CAPTURE(coeff5);
  const double coeff6 = coeff_dis(gen);
  CAPTURE(coeff6);
  const double coeff7 = coeff_dis(gen);
  CAPTURE(coeff7);
  const double coeff8 = coeff_dis(gen);
  CAPTURE(coeff8);
  const double coeff9 = coeff_dis(gen);
  CAPTURE(coeff9);
  const double coeff10 = coeff_dis(gen);
  CAPTURE(coeff10);
  const double coeff11 = coeff_dis(gen);
  CAPTURE(coeff11);
  const std::array<double, 4> coeffs = {coeff0, coeff1, coeff2, coeff3};
  const std::vector<std::array<double, 4>> datavector_coeffs = {
      {{coeff0, coeff1, coeff2, coeff3}},
      {{coeff4, coeff5, coeff6, coeff7}},
      {{coeff8, coeff9, coeff10, coeff11}}};
  const std::vector<double> vecx = {0.0, 1.0, 3.0, 2.0, 4.0,
                                    5.0, 6.0, 8.0, 7.0, 9.0};
  const DataVector datavecx = {0.0, 1.0, 2.0, 3.0, 4.0,
                               5.0, 8.0, 7.0, 6.0, 9.0};

  test_linear_least_squares_double<3>(coeffs, vecx);
  test_linear_least_squares_datavector();
  test_linear_least_squares_datavector2<3>(datavector_coeffs, datavecx);
}
