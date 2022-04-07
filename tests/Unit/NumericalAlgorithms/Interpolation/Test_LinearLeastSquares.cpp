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

namespace {
template <size_t Order>
void test_linear_least_squares(const std::array<double, Order + 1>& coeffs,
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

  const std::array<double, 4> coeffs = {coeff0, coeff1, coeff2, coeff3};
  const std::vector<double> vecx = {0.0, 1.0, 2.0, 3.0, 4.0,
                                    5.0, 6.0, 7.0, 8.0, 9.0};

  test_linear_least_squares<3>(coeffs, vecx);
}
