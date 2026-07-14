// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/UniformCardinalBSpline.hpp"

namespace {

template <typename F>
std::vector<double> sample_function(const F& function, const double start_time,
                                    const double time_step,
                                    const size_t num_samples) {
  std::vector<double> values(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    values[i] = function(start_time + static_cast<double>(i) * time_step);
  }
  return values;
}

// Maximum absolute deviation of the interpolant from the function over a
// dense sampling of the interpolant's bounds
template <typename F>
double max_error_against_function(
    const intrp::UniformCardinalBSpline& interpolant, const F& function) {
  const auto bounds = interpolant.bounds();
  const size_t num_test_points = 500;
  double max_error = 0.0;
  for (size_t i = 0; i <= num_test_points; ++i) {
    const double time = bounds[0] + (bounds[1] - bounds[0]) *
                                        static_cast<double>(i) /
                                        static_cast<double>(num_test_points);
    max_error =
        std::max(max_error, std::abs(interpolant(time) - function(time)));
  }
  return max_error;
}

void test_interpolation_and_convergence() {
  const auto function = [](const double time) { return std::sin(time); };
  const double start_time = 0.0;
  const double end_time = 2.0 * M_PI;

  const auto interpolant_error = [&function, &start_time,
                                  &end_time](const size_t num_samples) {
    const double time_step =
        (end_time - start_time) / static_cast<double>(num_samples - 1);
    // [uniform_cardinal_b_spline_example]
    const intrp::UniformCardinalBSpline interpolant{
        sample_function(function, start_time, time_step, num_samples),
        start_time, time_step};
    const double interpolated_value =
        interpolant(0.5 * (start_time + end_time));
    // [uniform_cardinal_b_spline_example]
    CHECK(interpolated_value ==
          approx(function(0.5 * (start_time + end_time))));
    return max_error_against_function(interpolant, function);
  };

  // The interpolation error should decrease with 4th order when the time step
  // is halved. Expect a factor of 16, check for at least 8.
  const double error_coarse = interpolant_error(17);
  const double error_fine = interpolant_error(33);
  CHECK(error_fine < error_coarse / 8.0);

  // The interpolant passes through the samples
  const size_t num_samples = 17;
  const double time_step =
      (end_time - start_time) / static_cast<double>(num_samples - 1);
  const auto values =
      sample_function(function, start_time, time_step, num_samples);
  const intrp::UniformCardinalBSpline interpolant{values, start_time,
                                                  time_step};
  for (size_t i = 0; i < num_samples; ++i) {
    CHECK(interpolant(start_time + static_cast<double>(i) * time_step) ==
          approx(values[i]));
  }

  // Accessors
  CHECK(interpolant.values() == values);
  CHECK(interpolant.start_time() == start_time);
  CHECK(interpolant.time_step() == time_step);
  CHECK(interpolant.bounds()[0] == approx(start_time));
  CHECK(interpolant.bounds()[1] == approx(end_time));

  // Evaluation at the exact bounds is clamped and does not throw
  CHECK(interpolant(interpolant.bounds()[0]) == approx(values.front()));
  CHECK(interpolant(interpolant.bounds()[1]) == approx(values.back()));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      interpolant(end_time + 1.0),
      Catch::Matchers::ContainsSubstring("is outside the sampled interval"));
#endif
}

void test_serialization() {
  const auto function = [](const double time) { return std::cos(time); };
  const intrp::UniformCardinalBSpline interpolant{
      sample_function(function, 1.2, 0.3, 20), 1.2, 0.3};
  const auto deserialized = serialize_and_deserialize(interpolant);
  CHECK(deserialized == interpolant);
  CHECK(deserialized(2.5) == interpolant(2.5));
  intrp::UniformCardinalBSpline copy{};
  copy = interpolant;
  CHECK(copy == interpolant);
  CHECK(copy(2.5) == interpolant(2.5));

  // Default-constructed (empty) interpolants can be serialized too
  const intrp::UniformCardinalBSpline default_constructed{};
  CHECK(default_constructed != interpolant);
  CHECK(serialize_and_deserialize(default_constructed) == default_constructed);
}

void test_estimate_interpolation_error() {
  const auto function = [](const double time) { return std::sin(time); };
  const double start_time = 0.0;
  const double end_time = 2.0 * M_PI;
  // Check both an odd and an even number of samples. For an even number the
  // coarser interpolant ends one time step early.
  for (const size_t num_samples : std::array<size_t, 2>{{33, 32}}) {
    CAPTURE(num_samples);
    const double time_step =
        (end_time - start_time) / static_cast<double>(num_samples - 1);
    const auto values =
        sample_function(function, start_time, time_step, num_samples);
    const double estimated_error =
        intrp::estimate_interpolation_error(values, start_time, time_step);
    const double actual_error = max_error_against_function(
        intrp::UniformCardinalBSpline{values, start_time, time_step}, function);
    // The estimate is conservative for smooth data, but not wildly so
    CHECK(estimated_error >= actual_error);
    CHECK(estimated_error <= 100.0 * actual_error);
  }

  CHECK_THROWS_WITH(
      intrp::estimate_interpolation_error({1.0, 2.0, 3.0, 4.0, 5.0}, 0.0, 1.0),
      Catch::Matchers::ContainsSubstring("At least 9 samples are required"));
}

void test_compress_to_tolerance() {
  const double start_time = 0.0;
  const double time_step = 4.0e-3;
  const size_t num_samples = 1001;

  {
    // Slowly varying data compresses to few points and stays within tolerance
    const auto function = [](const double time) {
      return std::sin(0.1 * time);
    };
    const auto values =
        sample_function(function, start_time, time_step, num_samples);
    const double tolerance = 1.0e-8;
    const auto [interpolant, max_error] =
        intrp::compress_to_tolerance(values, start_time, time_step, tolerance);
    CHECK(interpolant.values().size() < 100);
    CHECK(max_error <= tolerance);
    double max_deviation = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
      max_deviation =
          std::max(max_deviation,
                   std::abs(interpolant(start_time +
                                        static_cast<double>(i) * time_step) -
                            values[i]));
    }
    CHECK(max_deviation <= tolerance);
  }

  {
    // Linear data is reproduced exactly by the minimal number of points
    const auto function = [](const double time) { return 2.0 * time + 1.0; };
    const auto values =
        sample_function(function, start_time, time_step, num_samples);
    const auto [interpolant, max_error] =
        intrp::compress_to_tolerance(values, start_time, time_step, 1.0e-10);
    CHECK(interpolant.values().size() == 6);
    CHECK(max_error <= 1.0e-10);
  }

  {
    // Data that cannot be compressed is returned at full resolution with the
    // error of the last coarser candidate
    const auto function = [](const double time) {
      return std::sin(1.0e3 * time);
    };
    const auto values =
        sample_function(function, start_time, time_step, num_samples);
    const auto [interpolant, max_error] =
        intrp::compress_to_tolerance(values, start_time, time_step, 1.0e-300);
    CHECK(interpolant.values() == values);
    CHECK(max_error > 0.0);
  }

  {
    // Inputs with 6 or fewer samples are returned unchanged
    const std::vector<double> values{1.0, 2.0, 4.0, 8.0, 16.0};
    const auto [interpolant, max_error] =
        intrp::compress_to_tolerance(values, start_time, time_step, 1.0e-16);
    CHECK(interpolant.values() == values);
    CHECK(max_error == 0.0);
  }
}

void test_construction_errors() {
  CHECK_THROWS_WITH(
      (intrp::UniformCardinalBSpline{{1.0, 2.0, 3.0, 4.0}, 0.0, 1.0}),
      Catch::Matchers::ContainsSubstring("At least 5 samples are required"));
  CHECK_THROWS_WITH(
      (intrp::UniformCardinalBSpline{{1.0, 2.0, 3.0, 4.0, 5.0}, 0.0, 0.0}),
      Catch::Matchers::ContainsSubstring(
          "The time step must be finite and positive"));
  CHECK_THROWS_WITH(
      (intrp::UniformCardinalBSpline{{1.0, 2.0, 3.0, 4.0, 5.0}, 0.0, -1.0}),
      Catch::Matchers::ContainsSubstring(
          "The time step must be finite and positive"));
  CHECK_THROWS_WITH(
      (intrp::UniformCardinalBSpline{{1.0, 2.0, 3.0, 4.0, 5.0},
                                     std::numeric_limits<double>::infinity(),
                                     1.0}),
      Catch::Matchers::ContainsSubstring("The start time must be finite"));
  CHECK_THROWS_WITH(
      (intrp::UniformCardinalBSpline{
          {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0, 5.0},
          0.0,
          1.0}),
      Catch::Matchers::ContainsSubstring("The sampled values must be finite"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.UniformCardinalBSpline",
                  "[Unit][NumericalAlgorithms]") {
  test_interpolation_and_convergence();
  test_serialization();
  test_estimate_interpolation_error();
  test_compress_to_tolerance();
  test_construction_errors();
}
