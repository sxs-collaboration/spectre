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

namespace intrp {

template <typename Generator>
void test_linear_interpolator(const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 1.0};
  // cannot be const due to participation in the `span`
  auto linear_interpolator_values = make_with_random_values<DataVector>(
      gen, value_dist, static_cast<size_t>(2));
  auto linear_interpolator_complex_values =
      make_with_random_values<ComplexDataVector>(gen, value_dist,
                                                 static_cast<size_t>(2));
  const DataVector linear_interpolator_points = {{0.0, 1.0}};
  const double target_point = value_dist(*gen);
  const LinearSpanInterpolator test_linear_interpolator{};
  const double real_linear_interpolation = test_linear_interpolator.interpolate(
      gsl::span<const double>{linear_interpolator_points.data(),
                              linear_interpolator_points.size()},
      gsl::span<const double>{linear_interpolator_values.data(),
                              linear_interpolator_values.size()},
      target_point);
  CHECK(real_linear_interpolation ==
        approx(target_point * linear_interpolator_values[1] +
               (1.0 - target_point) * linear_interpolator_values[0]));
  const std::complex<double> complex_linear_interpolation =
      test_linear_interpolator.interpolate(
          gsl::span<const double>{linear_interpolator_points.data(),
                                  linear_interpolator_points.size()},
          gsl::span<const std::complex<double>>{
              linear_interpolator_complex_values.data(),
              linear_interpolator_complex_values.size()},
          target_point);
  CHECK_COMPLEX_APPROX(
      complex_linear_interpolation,
      target_point * linear_interpolator_complex_values[1] +
          (1.0 - target_point) * linear_interpolator_complex_values[0]);
}

template <typename VectorType, typename InterpolatorType, typename Generator>
void test_interpolator_approximate_fidelity(
    const gsl::not_null<Generator*> gen, const InterpolatorType& interpolator,
    Approx interpolator_approx) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 1.0};
  DataVector interpolator_points{
      2 * interpolator.required_number_of_points_before_and_after()};
  VectorType interpolator_values{
      2 * interpolator.required_number_of_points_before_and_after()};
  const double frequency = value_dist(*gen);
  const typename VectorType::value_type amplitude = value_dist(*gen);
  // only sample a small span to give the low-order interpolators an easier time
  for (size_t i = 0; i < interpolator_points.size(); ++i) {
    interpolator_points[i] = 0.01 * i / interpolator_points.size();
    interpolator_values[i] =
        amplitude * cos(frequency * interpolator_points[i]);
  }
  const double target_time = value_dist(*gen) / 100.0;
  const auto interpolator_result = interpolator.interpolate(
      gsl::span<const double>{interpolator_points.data(),
                              interpolator_points.size()},
      gsl::span<const typename VectorType::value_type>{
          interpolator_values.data(), interpolator_points.size()},
      target_time);

  CHECK_COMPLEX_CUSTOM_APPROX(interpolator_result,
                              amplitude * cos(frequency * target_time),
                              interpolator_approx);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolation.SpanInterpolators",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(gen);
  test_linear_interpolator(make_not_null(&gen));

  {
    // Linear interpolator will not get terribly close, but that's okay.
    Approx interpolator_approx = Approx::custom().epsilon(1.0e-2).scale(1.0);

    INFO("testing LinearSpanInterpolator")
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), LinearSpanInterpolator{}, interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), LinearSpanInterpolator{}, interpolator_approx);

    // verify the the construction from options is successful
    const auto option_created_linear_interpolator =
        TestHelpers::test_factory_creation<intrp::SpanInterpolator>(
            "LinearSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_linear_interpolator,
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_linear_interpolator,
        interpolator_approx);

    // verify that the interpolator can be serialized and deserialized
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(LinearSpanInterpolator{}),
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(LinearSpanInterpolator{}),
        interpolator_approx);
  }

  {
    Approx interpolator_approx =
        Approx::custom()
            .epsilon(std::numeric_limits<double>::epsilon() * 1.0e8)
            .scale(1.0);

    INFO("testing CubicSpanInterpolator")
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), CubicSpanInterpolator{}, interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), CubicSpanInterpolator{}, interpolator_approx);

    // verify the the construction from options is successful
    const auto option_created_cubic_interpolator =
        TestHelpers::test_factory_creation<intrp::SpanInterpolator>(
            "CubicSpanInterpolator");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_cubic_interpolator,
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_cubic_interpolator,
        interpolator_approx);

    // verify that the interpolator can be serialized and deserialized
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), serialize_and_deserialize(CubicSpanInterpolator{}),
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), serialize_and_deserialize(CubicSpanInterpolator{}),
        interpolator_approx);
  }

  {
    Approx interpolator_approx =
        Approx::custom()
            .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
            .scale(1.0);

    INFO("testing BarycentricRationalSpanInterpolator")
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u},
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), BarycentricRationalSpanInterpolator{5u, 6u},
        interpolator_approx);

    // verify the the construction from options is successful
    const auto option_created_barycentric_interpolator =
        TestHelpers::test_factory_creation<intrp::SpanInterpolator>(
            "BarycentricRationalSpanInterpolator:\n"
            "  MinOrder: 5\n"
            "  MaxOrder: 6");
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen), *option_created_barycentric_interpolator,
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen), *option_created_barycentric_interpolator,
        interpolator_approx);

    // verify that the interpolator can be serialized and deserialized
    test_interpolator_approximate_fidelity<DataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(BarycentricRationalSpanInterpolator{5u, 6u}),
        interpolator_approx);
    test_interpolator_approximate_fidelity<ComplexDataVector>(
        make_not_null(&gen),
        serialize_and_deserialize(BarycentricRationalSpanInterpolator{5u, 6u}),
        interpolator_approx);
  }
}
}  // namespace intrp
