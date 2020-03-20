// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <sharp_cxx.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "ErrorHandling/Error.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace Swsh {
namespace {

void test_swsh_coefficients_class_interface() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{8, 64};
  const size_t l_max = sdist(gen);

  CAPTURE(l_max);
  const CoefficientsMetadata& precomputed_libsharp_lm =
      cached_coefficients_metadata(l_max);

  const CoefficientsMetadata& another_precomputed_libsharp_lm =
      cached_coefficients_metadata(l_max);

  // checks that the same pointer is in both
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info() ==
        another_precomputed_libsharp_lm.get_sharp_alm_info());

  const CoefficientsMetadata computed_coefficients{l_max};

  CHECK(precomputed_libsharp_lm.l_max() == l_max);
  CHECK(computed_coefficients.l_max() == l_max);

  sharp_alm_info* expected_sharp_alm_info;
  sharp_make_triangular_alm_info(l_max, l_max, 1, &expected_sharp_alm_info);

  // check that all of the precomputed coefficients, the directly constructed
  // coefficients, and the manually created sharp_alm_info* all contain the same
  // data
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->lmax ==
        computed_coefficients.get_sharp_alm_info()->lmax);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->lmax ==
        expected_sharp_alm_info->lmax);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->lmax ==
        computed_coefficients.l_max());

  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->nm ==
        computed_coefficients.get_sharp_alm_info()->nm);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->nm ==
        expected_sharp_alm_info->nm);

  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->flags ==
        computed_coefficients.get_sharp_alm_info()->flags);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->flags ==
        expected_sharp_alm_info->flags);

  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->stride ==
        computed_coefficients.get_sharp_alm_info()->stride);
  CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->stride ==
        expected_sharp_alm_info->stride);

  for (size_t m_index = 0;
       m_index <
       static_cast<size_t>(precomputed_libsharp_lm.get_sharp_alm_info()->nm);
       ++m_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mval[m_index] ==
          computed_coefficients.get_sharp_alm_info()->mval[m_index]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mval[m_index] ==
          expected_sharp_alm_info->mval[m_index]);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mvstart[m_index] ==
          computed_coefficients.get_sharp_alm_info()->mvstart[m_index]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(precomputed_libsharp_lm.get_sharp_alm_info()->mvstart[m_index] ==
          expected_sharp_alm_info->mvstart[m_index]);
  }

  CHECK(precomputed_libsharp_lm.begin() == precomputed_libsharp_lm.cbegin());
  CHECK(precomputed_libsharp_lm.end() == precomputed_libsharp_lm.cend());
  CHECK(precomputed_libsharp_lm.begin() != precomputed_libsharp_lm.end());

  size_t offset_counter = 0;
  size_t expected_l = 0;
  size_t expected_m = 0;
  for (const auto& coefficient_info : precomputed_libsharp_lm) {
    CHECK(coefficient_info.transform_of_real_part_offset == offset_counter);
    CHECK(coefficient_info.transform_of_imag_part_offset -
              coefficient_info.transform_of_real_part_offset ==
          size_of_libsharp_coefficient_vector(l_max) / 2);
    CHECK(coefficient_info.l == expected_l);
    CHECK(coefficient_info.m == expected_m);
    ++offset_counter;
    if (expected_l == l_max) {
      ++expected_m;
      expected_l = expected_m;
    } else {
      ++expected_l;
    }
  }
}

template <int Spin, ComplexRepresentation Representation =
                        ComplexRepresentation::Interleaved>
void check_goldberg_mode_conversion() {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because Goldberg test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{4, 8};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  CAPTURE(l_max);
  const CoefficientsMetadata& precomputed_libsharp_lm =
      cached_coefficients_metadata(l_max);

  // low value to limit test time
  size_t number_of_radial_points = 2;

  ComplexModalVector expected_goldberg_modes =
      make_with_random_values<ComplexModalVector>(
          make_not_null(&gen), make_not_null(&coefficient_distribution),
          square(l_max + 1) * number_of_radial_points);

  // set to zero all modes for l < Spin (the basis functions are zero for those
  // modes)
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    for (size_t j = 0; j < static_cast<size_t>(square(Spin)); ++j) {
      expected_goldberg_modes[j + square(l_max + 1) * i] = 0.0;
    }
  }

  SpinWeighted<ComplexDataVector, Spin> goldberg_collocation_points;
  goldberg_collocation_points.data() = ComplexDataVector{
      number_of_radial_points * number_of_swsh_collocation_points(l_max), 0.0};

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    for (int l = 0; l <= static_cast<int>(l_max); ++l) {
      for (int m = -l; m <= l; ++m) {
        for (const auto& collocation_point :
             cached_collocation_metadata<Representation>(l_max)) {
          goldberg_collocation_points
              .data()[collocation_point.offset +
                      i * number_of_swsh_collocation_points(l_max)] +=
              expected_goldberg_modes[static_cast<size_t>(square(l) + m + l) +
                                      i * square(l_max + 1)] *
              TestHelpers::spin_weighted_spherical_harmonic(
                  Spin, l, m, collocation_point.theta, collocation_point.phi);
        }
      }
    }
  }

  SpinWeighted<ComplexModalVector, Spin> test_modes =
      swsh_transform<Representation>(l_max, number_of_radial_points,
                                     goldberg_collocation_points);

  Approx swsh_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  const auto goldberg_modes = libsharp_to_goldberg_modes(test_modes, l_max);
  CHECK_ITERABLE_CUSTOM_APPROX(goldberg_modes.data(), expected_goldberg_modes,
                               swsh_approx);

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    for (const auto& coefficient_info : precomputed_libsharp_lm) {
      auto goldberg_mode_plus_m =
          libsharp_mode_to_goldberg_plus_m(coefficient_info, test_modes, i);
      // should be the same value computed using the other interface
      auto alternative_mode_plus_m = libsharp_mode_to_goldberg(
          coefficient_info.l, static_cast<int>(coefficient_info.m), l_max,
          test_modes, i);
      auto goldberg_mode_plus_m_from_full_set =
          goldberg_modes.data()[goldberg_mode_index(
              l_max, coefficient_info.l, static_cast<int>(coefficient_info.m),
              i)];

      auto goldberg_mode_minus_m =
          libsharp_mode_to_goldberg_minus_m(coefficient_info, test_modes, i);
      // should be the same value computed using the other interface
      auto alternative_mode_minus_m = libsharp_mode_to_goldberg(
          coefficient_info.l, -static_cast<int>(coefficient_info.m), l_max,
          test_modes, i);
      auto goldberg_mode_minus_m_from_full_set =
          goldberg_modes.data()[goldberg_mode_index(
              l_max, coefficient_info.l, -static_cast<int>(coefficient_info.m),
              i)];

      CHECK_COMPLEX_CUSTOM_APPROX(goldberg_mode_plus_m, alternative_mode_plus_m,
                                  swsh_approx);
      CHECK_COMPLEX_CUSTOM_APPROX(goldberg_mode_plus_m,
                                  goldberg_mode_plus_m_from_full_set,
                                  swsh_approx);
      CHECK_COMPLEX_CUSTOM_APPROX(goldberg_mode_plus_m,
                                  expected_goldberg_modes[goldberg_mode_index(
                                      l_max, coefficient_info.l,
                                      static_cast<int>(coefficient_info.m), i)],
                                  swsh_approx);

      CHECK_COMPLEX_CUSTOM_APPROX(goldberg_mode_minus_m,
                                  alternative_mode_minus_m, swsh_approx);
      CHECK_COMPLEX_CUSTOM_APPROX(goldberg_mode_minus_m,
                                  goldberg_mode_minus_m_from_full_set,
                                  swsh_approx);
      CHECK_COMPLEX_CUSTOM_APPROX(
          goldberg_mode_minus_m,
          expected_goldberg_modes[goldberg_mode_index(
              l_max, coefficient_info.l, -static_cast<int>(coefficient_info.m),
              i)],
          swsh_approx);

      goldberg_modes_to_libsharp_modes_single_pair(
          coefficient_info, make_not_null(&test_modes), i,
          expected_goldberg_modes[goldberg_mode_index(
              l_max, coefficient_info.l, static_cast<int>(coefficient_info.m),
              i)],
          expected_goldberg_modes[goldberg_mode_index(
              l_max, coefficient_info.l, -static_cast<int>(coefficient_info.m),
              i)]);
    }
  }

  const auto inverse_transform_set_from_goldberg =
      inverse_swsh_transform<Representation>(l_max, number_of_radial_points,
                                             test_modes);
  CHECK_ITERABLE_CUSTOM_APPROX(inverse_transform_set_from_goldberg.data(),
                               goldberg_collocation_points.data(), swsh_approx);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshCoefficients",
                  "[Unit][NumericalAlgorithms]") {
  {
    INFO("Checking Coefficients for libsharp interoperability");
    test_swsh_coefficients_class_interface();
  }
  {
    INFO("Checking Goldberg mode conversions");
    check_goldberg_mode_conversion<-1>();
    check_goldberg_mode_conversion<0>();
    check_goldberg_mode_conversion<2>();
  }
}

// [[OutputRegex, Index out of range]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.SwshCoefficients.PrecomputationOverrun",
    "[Unit][NumericalAlgorithms]") {
  ERROR_TEST();
  cached_coefficients_metadata(detail::coefficients_maximum_l_max + 1);
  ERROR("Failed to trigger ERROR in an error test");
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
