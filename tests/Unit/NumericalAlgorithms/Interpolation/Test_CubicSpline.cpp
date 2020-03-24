// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"

namespace {
template <class F>
void test_cubic_spline(const F& function, const double lower_bound,
                       const double upper_bound, const size_t size,
                       const double tolerance,
                       const double tolerance_interior) noexcept {
  // Construct random points between lower and upper bound to interpolate
  // through. Always include the bounds in the x-values.
  std::vector<double> x_values(size), y_values(size);
  const double delta_x = (upper_bound - lower_bound) / size;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0., delta_x);
  x_values.front() = lower_bound;
  for (size_t i = 1; i < size - 1; ++i) {
    x_values[i] = lower_bound + i * delta_x + dist(gen);
  }
  x_values.back() = upper_bound;
  for (size_t i = 0; i < size; ++i) {
    y_values[i] = function(x_values[i]);
  }

  Approx custom_approx = Approx::custom().epsilon(tolerance).scale(1.);
  Approx custom_approx_interior =
      Approx::custom().epsilon(tolerance_interior).scale(1.);

  // Construct the interpolant and give an example
  /// [interpolate_example]
  intrp::CubicSpline interpolant{x_values, y_values};
  const double x_to_interpolate_to =
      lower_bound + (upper_bound - lower_bound) / 2.;
  CHECK(interpolant(x_to_interpolate_to) ==
        custom_approx_interior(function(x_to_interpolate_to)));
  /// [interpolate_example]

  // Check that the interpolation is exact at the datapoints
  for (const auto& x_value : x_values) {
    CHECK(interpolant(x_value) == approx(function(x_value)));
  }

  // Check that the interpolation matches the function within the given
  // tolerance. Also check that the serialized-and-deserialized interpolant does
  // the same.
  const auto deserialized_interpolant = serialize_and_deserialize(interpolant);
  double max_error = 0.;
  double max_error_x_value = std::numeric_limits<double>::signaling_NaN();
  double max_error_interior = 0.;
  double max_error_interior_x_value =
      std::numeric_limits<double>::signaling_NaN();
  for (size_t i = 0; i < 10 * size; ++i) {
    const double x_value = lower_bound + i * delta_x * 0.1 + 0.1 * dist(gen);
    CAPTURE(x_value);
    const double y_value = function(x_value);
    const double interpolated_y_value = interpolant(x_value);
    CHECK(interpolated_y_value == custom_approx(y_value));
    CHECK(deserialized_interpolant(x_value) == interpolated_y_value);
    // Record max error for better test failure reports
    const double error = abs(interpolated_y_value - y_value);
    if (error >= max_error) {
      max_error = error;
      max_error_x_value = x_value;
    }
    // Test the interpolation away from the boundaries with a lower tolerance.
    // Since this is a cubic spline interpolation, boundary effects should be
    // confined to the outer three interpolation points.
    const double boundary_fraction = 0.3;
    if (i > 10 * size * boundary_fraction and
        i < 10 * size * (1. - boundary_fraction)) {
      CHECK(interpolated_y_value == custom_approx_interior(y_value));
      if (error >= max_error_interior) {
        max_error_interior = error;
        max_error_interior_x_value = x_value;
      }
    }
  }
  // Output information on the precision the interpolation achieved when it
  // failed to stay within the given tolerances
  CAPTURE_PRECISE(max_error);
  CAPTURE_PRECISE(max_error_interior);
  // These checks are needed to trigger the max_error captures above
  CHECK(interpolant(max_error_x_value) ==
        custom_approx(function(max_error_x_value)));
  CHECK(interpolant(max_error_interior_x_value) ==
        custom_approx_interior(function(max_error_interior_x_value)));

  // Make sure moving the interpolant doesn't break anything
  const auto moved_interpolant = std::move(interpolant);
  const double x_value = lower_bound + dist(gen) * size;
  const double y_value = function(x_value);
  CHECK(moved_interpolant(x_value) == custom_approx(y_value));
}

void test_with_polynomial(const size_t number_of_points,
                          const size_t polynomial_degree,
                          const double tolerance,
                          const double tolerance_interior) {
  CAPTURE(polynomial_degree);
  CAPTURE(number_of_points);
  std::vector<double> coeffs(polynomial_degree + 1, 1.);
  test_cubic_spline(
      [&coeffs](const auto& x) noexcept {
        return evaluate_polynomial(coeffs, x);
      },
      -1., 2.3, number_of_points, tolerance, tolerance_interior);
}

void test_with_natural_boundary(const size_t number_of_points,
                                const double tolerance) {
  CAPTURE(number_of_points);
  // Precision at the boundaries should be the same as in the interior since
  // the natural boundary conditions are correct for this function
  test_cubic_spline(
      [](const auto& x) noexcept { return cube(sin(x)); }, 0., M_PI,
      number_of_points, tolerance, tolerance);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.CubicSpline",
                  "[Unit][NumericalAlgorithms]") {
  test_with_polynomial(10, 1, 1.e-14, 1.e-14);
  test_with_polynomial(10, 2, 1.e-1, 1.e-2);
  test_with_polynomial(100, 2, 1.e-3, 1.e-12);
  test_with_polynomial(100, 3, 1.e-2, 1.e-12);
  test_with_polynomial(1000, 3, 1.e-4, 1.e-12);
  test_with_polynomial(1000, 4, 1.e-3, 1.e-9);
  test_with_polynomial(1000, 5, 1.e-3, 1.e-8);
  test_with_natural_boundary(10, 1.e-1);
  test_with_natural_boundary(100, 1.e-5);
  test_with_natural_boundary(1000, 1.e-9);
}
