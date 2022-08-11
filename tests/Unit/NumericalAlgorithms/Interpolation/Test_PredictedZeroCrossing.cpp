// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <deque>
#include <random>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"
#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"

namespace {

void test_predicted_zero_crossing_datavector() {
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  std::uniform_real_distribution<> error_dist(-1e-8, 1e-8);

  DataVector expected_zero_crossing_value(3, 0.);
  for (size_t i = 0; i < expected_zero_crossing_value.size(); ++i) {
    expected_zero_crossing_value[i] = dist(gen);
  }
  CAPTURE(expected_zero_crossing_value);

  std::deque<double> slope{};
  for (size_t i = 0; i < expected_zero_crossing_value.size(); ++i) {
    slope.push_back(dist(gen));
  }
  CAPTURE(slope);

  std::deque<double> y_intercept{};
  for (size_t i = 0; i < expected_zero_crossing_value.size(); ++i) {
    y_intercept.push_back(-slope[i] * expected_zero_crossing_value[i]);
  }

  std::deque<double> x_values{0., -1., -2., -3., -4., -5., -6., -7., -8., -9.};
  std::deque<DataVector> y_values{};

  for (size_t i = 0; i < x_values.size(); i++) {
    DataVector tmp(expected_zero_crossing_value.size());
    for (size_t j = 0; j < tmp.size(); j++) {
      tmp[j] = y_intercept[j] + slope[j] * x_values[i] + error_dist(gen);
    }
    y_values.push_back(tmp);
  }

  DataVector compute_zero_crossing_value =
      intrp::predicted_zero_crossing_value(x_values, y_values);

  Approx custom_approx = Approx::custom().epsilon(1e-6);
  CHECK_ITERABLE_CUSTOM_APPROX(compute_zero_crossing_value,
                               expected_zero_crossing_value, custom_approx);
}

void test_predicted_zero_crossing_indeterminate() {
  // Here we set up points so that the error bars are large enough
  // that it is not clear whether the zero crossing is in the
  // past or in the future.
  std::vector<double> x_values{0.0, -1.0, -2.0, -3.0};
  std::vector<double> y_values{1.0, 2.1, 1.0, 2.0};
  CHECK(intrp::predicted_zero_crossing_value(x_values, y_values) == 0.0);
}

}  // namespace

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

  std::vector<double> x_values = {0.,  -1., -2., -3., -4.,
                                  -5., -6., -7., -8., -9.};
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

  // Test the zero crossing value for a set of datavectors
  test_predicted_zero_crossing_datavector();

  test_predicted_zero_crossing_indeterminate();
}
