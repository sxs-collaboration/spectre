// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace intrp {

template <typename Generator>
void test_linear_interpolator(const gsl::not_null<Generator*> gen) {
  UniformCustomDistribution<double> value_dist{0.1, 1.0};
  // cannot be const due to participation in the `span`
  auto linear_interpolator_values = make_with_random_values<DataVector>(
      gen, value_dist, static_cast<size_t>(2));
  auto linear_interpolator_complex_values =
      make_with_random_values<ComplexDataVector>(gen, value_dist,
                                                 static_cast<size_t>(2));
  // Use source points whose spacing is not 1 and that do not start at 0 so the
  // interpolation denominator and offset are genuinely exercised.
  const DataVector linear_interpolator_points = {{0.5, 2.0}};
  const double point_spacing =
      linear_interpolator_points[1] - linear_interpolator_points[0];
  const double target_point = value_dist(*gen);
  const LinearSpanInterpolator test_linear_interpolator{};
  const double real_linear_interpolation = test_linear_interpolator.interpolate(
      gsl::span<const double>{linear_interpolator_points.data(),
                              linear_interpolator_points.size()},
      gsl::span<const double>{linear_interpolator_values.data(),
                              linear_interpolator_values.size()},
      target_point);
  CHECK(real_linear_interpolation ==
        approx(linear_interpolator_values[0] +
               (linear_interpolator_values[1] - linear_interpolator_values[0]) /
                   point_spacing *
                   (target_point - linear_interpolator_points[0])));
  const std::complex<double> complex_linear_interpolation =
      test_linear_interpolator.interpolate(
          gsl::span<const double>{linear_interpolator_points.data(),
                                  linear_interpolator_points.size()},
          gsl::span<const std::complex<double>>{
              linear_interpolator_complex_values.data(),
              linear_interpolator_complex_values.size()},
          target_point);
  CHECK_ITERABLE_APPROX(
      complex_linear_interpolation,
      linear_interpolator_complex_values[0] +
          (linear_interpolator_complex_values[1] -
           linear_interpolator_complex_values[0]) /
              point_spacing * (target_point - linear_interpolator_points[0]));

  // The linear derivative is the constant slope between the two source
  // points; verify both the real and complex overloads.
  const double real_linear_derivative = test_linear_interpolator.derivative(
      gsl::span<const double>{linear_interpolator_points.data(),
                              linear_interpolator_points.size()},
      gsl::span<const double>{linear_interpolator_values.data(),
                              linear_interpolator_values.size()},
      target_point);
  CHECK(real_linear_derivative ==
        approx((linear_interpolator_values[1] - linear_interpolator_values[0]) /
               point_spacing));
  const std::complex<double> complex_linear_derivative =
      test_linear_interpolator.derivative(
          gsl::span<const double>{linear_interpolator_points.data(),
                                  linear_interpolator_points.size()},
          gsl::span<const std::complex<double>>{
              linear_interpolator_complex_values.data(),
              linear_interpolator_complex_values.size()},
          target_point);
  CHECK_ITERABLE_APPROX(complex_linear_derivative,
                        (linear_interpolator_complex_values[1] -
                         linear_interpolator_complex_values[0]) /
                            point_spacing);
}

template <typename VectorType, typename InterpolatorType, typename Generator>
void test_interpolator_approximate_fidelity(
    const gsl::not_null<Generator*> gen, const InterpolatorType& interpolator,
    Approx interpolator_approx) {
  UniformCustomDistribution<double> value_dist{0.1, 1.0};
  const size_t points_on_each_side =
      interpolator.required_number_of_points_before_and_after();
  DataVector interpolator_points{2 * points_on_each_side};
  VectorType interpolator_values{2 * points_on_each_side};
  const double frequency = value_dist(*gen);
  const typename VectorType::value_type amplitude = value_dist(*gen);
  // Sample a small, uniformly spaced span so the low-order interpolators stay
  // accurate.
  const double point_spacing = 0.01;
  for (size_t i = 0; i < interpolator_points.size(); ++i) {
    interpolator_points[i] = point_spacing * static_cast<double>(i);
    interpolator_values[i] =
        amplitude * cos(frequency * interpolator_points[i]);
  }
  // Keep the target in the central interval so the requested number of points
  // lies on each side of it (as required by the interpolator).
  const double target_time = interpolator_points[points_on_each_side - 1] +
                             value_dist(*gen) * point_spacing;
  const auto interpolator_result = interpolator.interpolate(
      gsl::span<const double>{interpolator_points.data(),
                              interpolator_points.size()},
      gsl::span<const typename VectorType::value_type>{
          interpolator_values.data(), interpolator_points.size()},
      target_time);

  CHECK_ITERABLE_CUSTOM_APPROX(interpolator_result,
                               amplitude * cos(frequency * target_time),
                               interpolator_approx);
}

template <typename VectorType, typename InterpolatorType, typename Generator>
void test_interpolator_derivative_approximate_fidelity(
    const gsl::not_null<Generator*> gen, const InterpolatorType& interpolator,
    Approx interpolator_approx) {
  // NOLINTNEXTLINE(misc-const-correctness) - operator() is non-const
  UniformCustomDistribution<double> value_dist{0.1, 1.0};
  const size_t points_on_each_side =
      interpolator.required_number_of_points_before_and_after();
  DataVector interpolator_points{2 * points_on_each_side};
  VectorType interpolator_values{2 * points_on_each_side};
  const double frequency = value_dist(*gen);
  const typename VectorType::value_type amplitude = value_dist(*gen);
  // Sample a small, uniformly spaced span so the low-order interpolators stay
  // accurate.
  const double point_spacing = 0.01;
  for (size_t i = 0; i < interpolator_points.size(); ++i) {
    interpolator_points[i] = point_spacing * static_cast<double>(i);
    interpolator_values[i] =
        amplitude * cos(frequency * interpolator_points[i]);
  }
  // Keep the target in the central interval so the requested number of points
  // lies on each side of it (as required by the interpolator).
  const double target_time = interpolator_points[points_on_each_side - 1] +
                             value_dist(*gen) * point_spacing;
  const auto interpolator_derivative_result = interpolator.derivative(
      gsl::span<const double>{interpolator_points.data(),
                              interpolator_points.size()},
      gsl::span<const typename VectorType::value_type>{
          interpolator_values.data(), interpolator_points.size()},
      target_time);

  CHECK_ITERABLE_CUSTOM_APPROX(
      interpolator_derivative_result,
      -amplitude * frequency * sin(frequency * target_time),
      interpolator_approx);
}

template <typename VectorType, typename InterpolatorType, typename Generator>
void test_interpolator_derivative_is_exact(
    const gsl::not_null<Generator*> gen, const InterpolatorType& interpolator) {
  // An order-N interpolant reproduces a degree-N polynomial exactly, so its
  // derivative equals the analytic polynomial derivative to roundoff (a small
  // multiple of machine epsilon). This is a much tighter accuracy check than
  // the fidelity test above, whose tolerance is dominated by the interpolant's
  // truncation error on a transcendental function.
  using ValueType = typename VectorType::value_type;
  const size_t number_of_points =
      2 * interpolator.required_number_of_points_before_and_after();
  const size_t degree = number_of_points - 1;
  const UniformCustomDistribution<double> coefficient_dist{0.1, 1.0};
  const auto coefficients = make_with_random_values<VectorType>(
      gen, coefficient_dist, number_of_points);
  // Horner evaluation of the polynomial and of its analytic derivative.
  const auto polynomial = [&coefficients, degree](const double x) {
    ValueType result = coefficients[degree];
    for (size_t i = degree; i > 0; --i) {
      result = result * x + coefficients[i - 1];
    }
    return result;
  };
  const auto polynomial_derivative = [&coefficients, degree](const double x) {
    ValueType result = static_cast<double>(degree) * coefficients[degree];
    // loop on `i > 1` and index with `i - 1` so the size_t cannot underflow
    // for a degree-0 polynomial.
    for (size_t i = degree; i > 1; --i) {
      result = result * x + static_cast<double>(i - 1) * coefficients[i - 1];
    }
    return result;
  };
  // Place the points near integer positions but randomly perturbed, so the
  // test does not rely on a special exactly-uniform grid. The perturbation
  // stays well below the unit spacing, keeping the points well separated and
  // the interpolation well conditioned.
  // NOLINTNEXTLINE(misc-const-correctness) - operator() is non-const
  UniformCustomDistribution<double> perturbation_dist{-0.1, 0.1};
  DataVector points{number_of_points};
  VectorType values{number_of_points};
  for (size_t i = 0; i < number_of_points; ++i) {
    points[i] = static_cast<double>(i) + perturbation_dist(*gen);
    values[i] = polynomial(points[i]);
  }
  const double target_point = 0.5 * static_cast<double>(degree);
  const auto interpolator_derivative_result = interpolator.derivative(
      gsl::span<const double>{points.data(), points.size()},
      gsl::span<const ValueType>{values.data(), values.size()}, target_point);
  const Approx exact_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(interpolator_derivative_result,
                               polynomial_derivative(target_point),
                               exact_approx);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolation.SpanInterpolators",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(gen);
  test_linear_interpolator(make_not_null(&gen));

  {
    // Over the small sampled span the test function is nearly linear, so even
    // the linear interpolator resolves the value well; its derivative is a
    // finite-difference slope that loses one factor of h in accuracy.
    const Approx interpolator_approx =
        Approx::custom().epsilon(1.0e-4).scale(1.0);
    const Approx derivative_approx =
        Approx::custom().epsilon(1.0e-2).scale(1.0);

    INFO("testing LinearSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), LinearSpanInterpolator{}, interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), LinearSpanInterpolator{}, interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), LinearSpanInterpolator{}, derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), LinearSpanInterpolator{}, derivative_approx);
    // a linear interpolant reproduces a degree-1 polynomial exactly, so its
    // derivative is exact -- a tight check the fidelity test cannot provide
    test_interpolator_derivative_is_exact<DataVector>(make_not_null(&gen),
                                                      LinearSpanInterpolator{});
    test_interpolator_derivative_is_exact<ComplexDataVector>(
        make_not_null(&gen), LinearSpanInterpolator{});

    // verify the the construction from options is successful
    const auto option_created_linear_interpolator =
        TestHelpers::test_creation<std::unique_ptr<intrp::SpanInterpolator>>(
            "LinearSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_linear_interpolator,
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_linear_interpolator,
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_linear_interpolator,
        derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_linear_interpolator,
        derivative_approx);

    // verify that the interpolator can be serialized and deserialized
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(LinearSpanInterpolator{}),
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(LinearSpanInterpolator{}),
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(LinearSpanInterpolator{}), derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(LinearSpanInterpolator{}), derivative_approx);
  }

  {
    const Approx interpolator_approx =
        Approx::custom()
            .epsilon(std::numeric_limits<double>::epsilon() * 1.0e8)
            .scale(1.0);
    // The cubic-Lagrange derivative loses one factor of h in accuracy
    // compared to the value of the interpolant.
    const Approx derivative_approx =
        Approx::custom().epsilon(1.0e-5).scale(1.0);

    INFO("testing CubicSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), CubicSpanInterpolator{}, interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), CubicSpanInterpolator{}, interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), CubicSpanInterpolator{}, derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), CubicSpanInterpolator{}, derivative_approx);
    // a cubic interpolant reproduces a cubic exactly, so its derivative is also
    // exact -- a tight check the fidelity test above cannot provide
    test_interpolator_derivative_is_exact<DataVector>(make_not_null(&gen),
                                                      CubicSpanInterpolator{});
    test_interpolator_derivative_is_exact<ComplexDataVector>(
        make_not_null(&gen), CubicSpanInterpolator{});

    // verify the the construction from options is successful
    const auto option_created_cubic_interpolator =
        TestHelpers::test_creation<std::unique_ptr<intrp::SpanInterpolator>>(
            "CubicSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_cubic_interpolator,
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_cubic_interpolator,
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_cubic_interpolator,
        derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_cubic_interpolator,
        derivative_approx);

    // verify that the interpolator can be serialized and deserialized
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), serialize_and_deserialize(CubicSpanInterpolator{}),
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), serialize_and_deserialize(CubicSpanInterpolator{}),
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), serialize_and_deserialize(CubicSpanInterpolator{}),
        derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), serialize_and_deserialize(CubicSpanInterpolator{}),
        derivative_approx);
  }

  {
    const Approx interpolator_approx =
        Approx::custom()
            .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
            .scale(1.0);
    // The barycentric-rational derivative is one order less accurate than the
    // value of the interpolant.
    const Approx derivative_approx =
        Approx::custom().epsilon(1.0e-8).scale(1.0);

    INFO("testing BarycentricRationalSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u},
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u},
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u},
        derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u},
        derivative_approx);
    // with 6 points and order 5 the interpolant reproduces a degree-5
    // polynomial exactly, so its derivative is exact as well
    test_interpolator_derivative_is_exact<DataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u});
    test_interpolator_derivative_is_exact<ComplexDataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u});

    // verify the the construction from options is successful
    const auto option_created_barycentric_interpolator =
        TestHelpers::test_creation<std::unique_ptr<intrp::SpanInterpolator>>(
            "BarycentricRationalSpanInterpolator:\n"
            "  MinOrder: 5\n"
            "  MaxOrder: 6");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_barycentric_interpolator,
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_barycentric_interpolator,
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_barycentric_interpolator,
        derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_barycentric_interpolator,
        derivative_approx);

    // verify that the interpolator can be serialized and deserialized
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(BarycentricRationalSpanInterpolator{5u, 6u}),
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(BarycentricRationalSpanInterpolator{5u, 6u}),
        interpolator_approx);
    test_interpolator_derivative_approximate_fidelity<DataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(BarycentricRationalSpanInterpolator{5u, 6u}),
        derivative_approx);
    test_interpolator_derivative_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(BarycentricRationalSpanInterpolator{5u, 6u}),
        derivative_approx);
  }
}
}  // namespace intrp
