// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>

#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/Literals.hpp"

namespace Spectral::Swsh {
namespace {

// [[OutputRegex, Attempting to perform interpolation]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.SwshInterpolation.InterpolateError",
    "[Unit][NumericalAlgorithms]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  SwshInterpolator interp{};
  SpinWeighted<ComplexDataVector, 1> interp_source{
      number_of_swsh_collocation_points(5_st)};
  SpinWeighted<ComplexDataVector, 1> interp_target{
      number_of_swsh_collocation_points(5_st)};
  interp.interpolate(make_not_null(&interp_target), interp_source);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Attempting to perform spin-weighted evaluation]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.SwshInterpolation.EvaluationError",
    "[Unit][NumericalAlgorithms]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  SwshInterpolator interp{};
  SpinWeighted<ComplexDataVector, 1> interp_target{
      number_of_swsh_collocation_points(5_st)};
  interp.direct_evaluation_swsh_at_l_min(make_not_null(&interp_target), 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

template <typename Generator>
void test_basis_function(const gsl::not_null<Generator*> generator) {
  UniformCustomDistribution<double> phi_dist{0.0, 2.0 * M_PI};
  const double phi = phi_dist(*generator);
  UniformCustomDistribution<double> theta_dist{0.01, M_PI - 0.01};
  const double theta = theta_dist(*generator);
  UniformCustomDistribution<int> spin_dist{-2, 2};
  const int spin = spin_dist(*generator);
  UniformCustomDistribution<size_t> l_dist{static_cast<size_t>(abs(spin)), 16};
  const size_t l = l_dist(*generator);
  UniformCustomDistribution<int> m_dist{-static_cast<int>(l),
                                        static_cast<int>(l)};
  const int m = m_dist(*generator);

  std::complex<double> expected = TestHelpers::spin_weighted_spherical_harmonic(
      spin, static_cast<int>(l), m, theta, phi);
  const auto test_harmonic = SpinWeightedSphericalHarmonic{spin, l, m};
  std::complex<double> test = test_harmonic.evaluate(theta, phi);
  CAPTURE(spin);
  CAPTURE(l);
  CAPTURE(m);
  CAPTURE(theta);
  CAPTURE(phi);

  // need a slightly loose approx to accommodate the explicit factorials in the
  // simpler TestHelper form
  Approx factorial_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(test, expected, factorial_approx);
  const auto deserialized_test_harmonic =
      serialize_and_deserialize(test_harmonic);
  test = deserialized_test_harmonic.evaluate(theta, phi);
  CHECK_ITERABLE_CUSTOM_APPROX(test, expected, factorial_approx);
}

template <int spin, typename Generator>
void test_interpolation(const gsl::not_null<Generator*> generator) {
  INFO("Testing interpolation for spin " << spin);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t l_max = 16;
  UniformCustomDistribution<double> phi_dist{0.0, 2.0 * M_PI};
  UniformCustomDistribution<double> theta_dist{0.01, M_PI - 0.01};
  const size_t number_of_target_points = 10;

  const auto target_phi = make_with_random_values<DataVector>(
      generator, make_not_null(&phi_dist), number_of_target_points);
  const auto target_theta = make_with_random_values<DataVector>(
      generator, make_not_null(&theta_dist), number_of_target_points);

  SpinWeighted<ComplexModalVector, spin> generated_modes{
      size_of_libsharp_coefficient_vector(l_max)};
  TestHelpers::generate_swsh_modes<spin>(
      make_not_null(&generated_modes.data()), generator,
      make_not_null(&coefficient_distribution), 1, l_max);

  const auto generated_collocation =
      inverse_swsh_transform(l_max, 1, generated_modes);

  const auto goldberg_modes =
      libsharp_to_goldberg_modes(generated_modes, l_max);

  SpinWeighted<ComplexDataVector, spin> expected{number_of_target_points, 0.0};
  SpinWeighted<ComplexDataVector, spin> another_expected{
      number_of_target_points, 0.0};
  auto interpolator = SwshInterpolator{target_theta, target_phi, l_max};
  const auto deserialized_interpolator =
      serialize_and_deserialize(interpolator);
  for (int l = 0; l <= static_cast<int>(l_max); ++l) {
    for(int m = -l; m <= l; ++m) {
      auto sYlm =
          SpinWeightedSphericalHarmonic{spin, static_cast<size_t>(l), m};
      if (l == std::max(abs(m), abs(spin))) {
        SpinWeighted<ComplexDataVector, spin> harmonic_test;
        interpolator.direct_evaluation_swsh_at_l_min(
            make_not_null(&harmonic_test), m);
        for(size_t i = 0; i < number_of_target_points; ++i) {
          CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                harmonic_test.data()[i]);
        }
      }
      if (l == std::max(abs(m), abs(spin)) + 1) {
        SpinWeighted<ComplexDataVector, spin> harmonic_test_l_min;
        interpolator.direct_evaluation_swsh_at_l_min(
            make_not_null(&harmonic_test_l_min), m);

        SpinWeighted<ComplexDataVector, spin> harmonic_test_l_min_plus_one;
        interpolator.evaluate_swsh_at_l_min_plus_one(
            make_not_null(&harmonic_test_l_min_plus_one), harmonic_test_l_min,
            m);

        for(size_t i = 0; i < number_of_target_points; ++i) {
          CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                harmonic_test_l_min_plus_one.data()[i]);
        }
      }
      if (l == std::max(abs(m), abs(spin)) and abs(m) > abs(spin)) {
        if(m > 0) {
          SpinWeighted<ComplexDataVector, spin> harmonic_test;
          interpolator.direct_evaluation_swsh_at_l_min(
              make_not_null(&harmonic_test), m - 1);
          interpolator.evaluate_swsh_m_recurrence_at_l_min(
              make_not_null(&harmonic_test), m);
          INFO("checking l=" << l <<" m=" << m);
          for(size_t i = 0; i < number_of_target_points; ++i) {
            CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                  harmonic_test.data()[i]);
          }

          // check the serialization hasn't harmed the interpolator
          deserialized_interpolator.direct_evaluation_swsh_at_l_min(
              make_not_null(&harmonic_test), m - 1);
          deserialized_interpolator.evaluate_swsh_m_recurrence_at_l_min(
              make_not_null(&harmonic_test), m);
          INFO("checking l=" << l << " m=" << m);
          for(size_t i = 0; i < number_of_target_points; ++i) {
            CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                  harmonic_test.data()[i]);
          }

        } else {
          SpinWeighted<ComplexDataVector, spin> harmonic_test;
          interpolator.direct_evaluation_swsh_at_l_min(
              make_not_null(&harmonic_test), m + 1);
          interpolator.evaluate_swsh_m_recurrence_at_l_min(
              make_not_null(&harmonic_test), m);
          INFO("checking l=" << l <<" m=" << m);
          for(size_t i = 0; i < number_of_target_points; ++i) {
            CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                  harmonic_test.data()[i]);
          }

          // check the serialization hasn't harmed the interpolator
          deserialized_interpolator.direct_evaluation_swsh_at_l_min(
              make_not_null(&harmonic_test), m + 1);
          deserialized_interpolator.evaluate_swsh_m_recurrence_at_l_min(
              make_not_null(&harmonic_test), m);
          INFO("checking l=" << l <<" m=" << m);
          for(size_t i = 0; i < number_of_target_points; ++i) {
            CHECK_ITERABLE_APPROX(sYlm.evaluate(target_theta[i], target_phi[i]),
                                  harmonic_test.data()[i]);
          }
        }
      }
      for(size_t i = 0; i < number_of_target_points; ++i) {
        // gcc warns about the casts in ways that are impossible to satisfy
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
        expected.data()[i] +=
            goldberg_modes.data()[
                square(static_cast<size_t>(l)) + static_cast<size_t>(l + m)] *
            TestHelpers::spin_weighted_spherical_harmonic(
                spin, l, m, target_theta[i], target_phi[i]);
        another_expected.data()[i] +=
            goldberg_modes.data()[square(static_cast<size_t>(l)) +
                                  static_cast<size_t>(l + m)] *
            sYlm.evaluate(target_theta[i], target_phi[i]);
      }
    }
  }

  Approx factorial_approx =
      Approx::custom()
      .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
      .scale(1.0);
  // direct test Clenshaw sums
  for(int m = -static_cast<int>(l_max); m <= static_cast<int>(l_max); ++m) {
    SpinWeighted<ComplexDataVector, spin> expected_clenshaw_sum{
        number_of_target_points, 0.0};
    for (int l = std::max(abs(m), abs(spin)); l <= static_cast<int>(l_max);
         ++l) {
      auto sYlm =
          SpinWeightedSphericalHarmonic{spin, static_cast<size_t>(l), m};
      for(size_t i = 0; i < number_of_target_points; ++i) {
        expected_clenshaw_sum.data()[i] +=
            goldberg_modes.data()[square(static_cast<size_t>(l)) +
                                  static_cast<size_t>(l + m)] *
            sYlm.evaluate(target_theta[i], target_phi[i]);
      }
    }
#pragma GCC diagnostic pop
    SpinWeighted<ComplexDataVector, spin> clenshaw{number_of_target_points,
                                                   0.0};

    SpinWeighted<ComplexDataVector, spin> harmonic_test_l_min;
    interpolator.direct_evaluation_swsh_at_l_min(
        make_not_null(&harmonic_test_l_min), m);

    SpinWeighted<ComplexDataVector, spin> harmonic_test_l_min_plus_one;
    interpolator.evaluate_swsh_at_l_min_plus_one(
        make_not_null(&harmonic_test_l_min_plus_one), harmonic_test_l_min, m);

    interpolator.clenshaw_sum(
        make_not_null(&clenshaw), harmonic_test_l_min,
        harmonic_test_l_min_plus_one,
        libsharp_to_goldberg_modes(
            swsh_transform(l_max, 1, generated_collocation), l_max),
        m);
    INFO("checking clenshaw sum for m=" << m);
    for(size_t i = 0; i < number_of_target_points; ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(clenshaw.data()[i],
                                   expected_clenshaw_sum.data()[i],
                                   factorial_approx);
    }
  }

  SpinWeighted<ComplexDataVector, spin> clenshaw_interpolation;
  interpolator.interpolate(
      make_not_null(&clenshaw_interpolation),
      libsharp_to_goldberg_modes(
          swsh_transform(l_max, 1, generated_collocation), l_max));

  CHECK_ITERABLE_CUSTOM_APPROX(expected, another_expected, factorial_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(clenshaw_interpolation, expected,
                               factorial_approx);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshInterpolation",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  // test a few points on each run of the test, these are cheap.
  for(size_t i = 0; i < 10; ++i) {
    test_basis_function(make_not_null(&generator));
  }

  test_interpolation<-1>(make_not_null(&generator));
  test_interpolation<0>(make_not_null(&generator));
  test_interpolation<2>(make_not_null(&generator));
}
}  // namespace
}  // namespace Spectral::Swsh
