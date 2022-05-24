// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"
#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolation.PredictedZeroCrossing",
    "[Unit][NumericalAlgorithms]") {
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  std::uniform_real_distribution<> error_dist(-1e-8, 1e-8);

  const double expected_zero_crossing_value = dist(gen);
  CAPTURE(expected_zero_crossing_value);
  const double slope = dist(gen);
  CAPTURE(slope);

  std::vector<double> x_values = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  std::vector<double> y_values{};
  double b = -slope * expected_zero_crossing_value;
  for (size_t i = 0; i < x_values.size(); i++) {
    double epsilon = error_dist(gen);
    y_values.push_back(slope * gsl::at(x_values, i) + b + epsilon);
  }

  auto compute_zero_crossing_value =
      intrp::predicted_zero_crossing_value(x_values, y_values);

  Approx custom_approx = Approx::custom().epsilon(1.e-6);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_zero_crossing_value,
                               compute_zero_crossing_value, custom_approx);
}
