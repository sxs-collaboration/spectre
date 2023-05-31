// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"

namespace {

void test_errors() {
  CHECK_THROWS_WITH(intrp::ZeroCrossingPredictor(2, 3),
                    Catch::Matchers::Contains("min_size must be >= 3,"));
  CHECK_THROWS_WITH(intrp::ZeroCrossingPredictor(6, 5),
                    Catch::Matchers::Contains("min_size must be <= max_size,"));
  CHECK_THROWS_WITH(intrp::ZeroCrossingPredictor(4, 5).zero_crossing_time(0.0),
                    Catch::Matchers::Contains("Invalid ZeroCrossingPredictor"));
}

void test_zero_crossing_predictor() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  std::uniform_real_distribution<> error_dist(-1e-8, 1e-8);

  const double expected_zero_crossing_value1 = dist(gen);
  CAPTURE(expected_zero_crossing_value1);
  const double expected_zero_crossing_value2 = dist(gen);
  CAPTURE(expected_zero_crossing_value2);
  const double slope1 = dist(gen);
  CAPTURE(slope1);
  const double slope2 = dist(gen);
  CAPTURE(slope2);

  std::vector<double> x_values = {-9., -8., -7., -6., -5.,
                                  -4., -3., -2., -1., 0.};
  std::vector<DataVector> y_values{};
  const double intercept1 = -slope1 * expected_zero_crossing_value1;
  const double intercept2 = -slope2 * expected_zero_crossing_value2;
  for (size_t i = 0; i < x_values.size(); i++) {
    double epsilon = error_dist(gen);
    y_values.push_back({slope1 * gsl::at(x_values, i) + intercept1 + epsilon,
                        slope2 * gsl::at(x_values, i) + intercept2 + epsilon});
  }

  constexpr size_t min_size = 4;
  intrp::ZeroCrossingPredictor predictor(min_size, x_values.size());

  // Check trivial case that min_positive_zero_crossing_time returns
  // zero for an invalid predictor.
  CHECK(predictor.min_positive_zero_crossing_time(x_values.back()) == 0.0);

  // Fill points in predictor.
  for (size_t i = 0; i < x_values.size(); i++) {
    predictor.add(x_values[i], y_values[i]);
    if (i < min_size - 1) {
      CHECK_FALSE(predictor.is_valid());
    } else {
      CHECK(predictor.is_valid());
    }
  }

  // Check serialization and operator== and operator!=
  CHECK(predictor == serialize_and_deserialize(predictor));
  CHECK_FALSE(predictor != serialize_and_deserialize(predictor));

  test_copy_semantics(predictor);
  test_move_semantics(serialize_and_deserialize(predictor),
                      serialize_and_deserialize(predictor), min_size,
                      x_values.size());

  Approx custom_approx = Approx::custom().epsilon(1e-5).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      predictor.zero_crossing_time(x_values.back()),
      SINGLE_ARG(DataVector{expected_zero_crossing_value1,
                            expected_zero_crossing_value2}),
      custom_approx);

  const double adjusted_value1 = expected_zero_crossing_value1 < 0.0
                                     ? std::numeric_limits<double>::infinity()
                                     : expected_zero_crossing_value1;
  const double adjusted_value2 = expected_zero_crossing_value2 < 0.0
                                     ? std::numeric_limits<double>::infinity()
                                     : expected_zero_crossing_value2;
  const double expected_zero_crossing =
      std::min(adjusted_value1, adjusted_value2);

  CHECK(predictor.min_positive_zero_crossing_time(x_values.back()) ==
        custom_approx(expected_zero_crossing));

  // Re-add the first point.  Adding the first point should cause the
  // (original) first point to be popped off the deque [and thus test
  // this popping], so that the result should be identical to the
  // previous result.
  DataVector first_point = y_values.front();
  predictor.add(x_values.front(), std::move(first_point));
  CHECK(predictor.min_positive_zero_crossing_time(0.0) ==
        custom_approx(expected_zero_crossing));

  // Clear the predictor, and check that min_positive_zero_crossing_time
  // returns zero again for the cleared (and now invalid) predictor.
  predictor.clear();
  CHECK(predictor.min_positive_zero_crossing_time(x_values.back()) == 0.0);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolation.ZeroCrossingPredictor",
    "[Unit][NumericalAlgorithms]") {
  test_zero_crossing_predictor();
  test_errors();
}
