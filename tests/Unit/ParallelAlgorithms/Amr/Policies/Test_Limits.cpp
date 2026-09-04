// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"

namespace {
constexpr size_t p_max = Spectral::limits::max_i1_polynomial_mode;
constexpr size_t l_min = Spectral::limits::min_spherical_harmonic_mode;
constexpr size_t l_max = Spectral::limits::max_spherical_harmonic_mode;
constexpr size_t m_max = Spectral::limits::max_fourier_mode;

void test_equality() {
  INFO("Equality");
  const amr::Limits limits{};
  CHECK(limits ==
        amr::Limits{
            {{0, 15}}, {{0, p_max}}, {{0, m_max}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{
          {{4, 15}}, {{0, p_max}}, {{0, m_max}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{
          {{0, 8}}, {{0, p_max}}, {{0, m_max}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{
          {{0, 15}}, {{6, p_max}}, {{0, m_max}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{{{0, 15}}, {{0, 6}}, {{0, m_max}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{
          {{0, 15}}, {{0, p_max}}, {{6, m_max}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{{{0, 15}}, {{0, p_max}}, {{0, 6}}, {{l_min, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{{{0, 15}}, {{0, p_max}}, {{0, m_max}}, {{6, l_max}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{{{0, 15}}, {{0, p_max}}, {{0, m_max}}, {{l_min, 6}}, false});
  CHECK_FALSE(
      limits ==
      amr::Limits{
          {{0, 15}}, {{0, p_max}}, {{0, m_max}}, {{l_min, l_max}}, true});
}

void test_pup() {
  INFO("Serialization");
  test_serialization(amr::Limits{});
  test_serialization(amr::Limits{{{0, 5}}, {{0, 4}}, {{0, 6}}, {{4, 7}}, true});
}

void test_option_parsing() {
  INFO("Option Parsing creation");
  {
    const std::string creation_string_1 =
        "RefinementLevel: [0, 3]\n"
        "NumPolynomialModes: [1, 5]\n"
        "FourierM: [3, 9]\n"
        "SphericalHarmonicL: [4, 8]\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_1);
    CHECK(limits == amr::Limits{{{0, 3}}, {{1, 5}}, {{3, 9}}, {{4, 8}}, false});
  }

  {
    const std::string creation_string_2 =
        "RefinementLevel: Auto\n"
        "NumPolynomialModes: [1, 5]\n"
        "FourierM: [3, 9]\n"
        "SphericalHarmonicL: [4, 8]\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_2);
    CHECK(limits == amr::Limits{{{0, ElementId<1>::max_refinement_level}},
                                {{1, 5}},
                                {{3, 9}},
                                {{4, 8}},
                                false});
  }

  {
    const std::string creation_string_3 =
        "RefinementLevel: [0, 3]\n"
        "NumPolynomialModes: Auto\n"
        "FourierM: [3, 9]\n"
        "SphericalHarmonicL: [4, 8]\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_3);
    CHECK(limits == amr::Limits{{{0, 3}},
                                {{0, Spectral::limits::max_i1_polynomial_mode}},
                                {{3, 9}},
                                {{4, 8}},
                                false});
  }

  {
    const std::string creation_string_4 =
        "RefinementLevel: [0, 3]\n"
        "NumPolynomialModes: [3, 9]\n"
        "FourierM: Auto\n"
        "SphericalHarmonicL: [4, 8]\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_4);
    CHECK(limits == amr::Limits{{{0, 3}},
                                {{3, 9}},
                                {{0, Spectral::limits::max_fourier_mode}},
                                {{4, 8}},
                                false});
  }

  {
    const std::string creation_string_5 =
        "RefinementLevel: [0, 3]\n"
        "NumPolynomialModes: [3, 9]\n"
        "FourierM: [4, 8]\n"
        "SphericalHarmonicL: Auto\n"
        "ErrorBeyondLimits: True\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_5);
    CHECK(limits ==
          amr::Limits{{{0, 3}},
                      {{3, 9}},
                      {{4, 8}},
                      {{Spectral::limits::min_spherical_harmonic_mode,
                        Spectral::limits::max_spherical_harmonic_mode}},
                      true});
  }

  {
    const std::string creation_string_6 =
        "RefinementLevel: Auto\n"
        "NumPolynomialModes: Auto\n"
        "FourierM: Auto\n"
        "SphericalHarmonicL: Auto\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_6);
    CHECK(limits == amr::Limits{});
  }

  const std::string bad_creation_string_1 =
      "RefinementLevel: [255, 3]\n"
      "NumPolynomialModes: Auto\n"
      "FourierM: Auto\n"
      "SphericalHarmonicL: Auto\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_1),
      Catch::Matchers::ContainsSubstring("RefinementLevel lower bound '255' "
                                         "cannot be larger than upper bound"));

  const std::string bad_creation_string_2 =
      "RefinementLevel: [3, 255]\n"
      "NumPolynomialModes: Auto\n"
      "FourierM: Auto\n"
      "SphericalHarmonicL: Auto\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_2),
      Catch::Matchers::ContainsSubstring(
          "RefinementLevel upper bound '255' "
          "cannot be larger than refinement limit"));

  const std::string bad_creation_string_3 =
      "RefinementLevel: Auto\n"
      "NumPolynomialModes: [255, 3]\n"
      "FourierM: Auto\n"
      "SphericalHarmonicL: Auto\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_3),
      Catch::Matchers::ContainsSubstring("NumPolynomialModes lower bound '255' "
                                         "cannot be larger than upper bound"));

  const std::string bad_creation_string_4 =
      "RefinementLevel: Auto\n"
      "NumPolynomialModes: [3, 255]\n"
      "FourierM: Auto\n"
      "SphericalHarmonicL: Auto\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_4),
      Catch::Matchers::ContainsSubstring(
          "NumPolynomialModes upper bound '255' "
          "cannot be larger than Spectral::limits::max_i1_polynomial_mode"));

  const std::string bad_creation_string_5 =
      "RefinementLevel: Auto\n"
      "NumPolynomialModes: Auto\n"
      "FourierM: [255, 3]\n"
      "SphericalHarmonicL: Auto\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_5),
      Catch::Matchers::ContainsSubstring("FourierM lower bound '255' "
                                         "cannot be larger than upper bound"));

  const std::string bad_creation_string_6 =
      "RefinementLevel: Auto\n"
      "NumPolynomialModes: Auto\n"
      "FourierM: [3, 255]\n"
      "SphericalHarmonicL: Auto\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_6),
      Catch::Matchers::ContainsSubstring(
          "FourierM upper bound '255' "
          "cannot be larger than Spectral::limits::max_fourier_mode"));

  const std::string bad_creation_string_7 =
      "RefinementLevel: Auto\n"
      "NumPolynomialModes: Auto\n"
      "FourierM: Auto\n"
      "SphericalHarmonicL: [255, 3]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_7),
      Catch::Matchers::ContainsSubstring("SphericalHarmonicL lower bound '255' "
                                         "cannot be larger than upper bound"));

  const std::string bad_creation_string_8 =
      "RefinementLevel: Auto\n"
      "NumPolynomialModes: Auto\n"
      "FourierM: Auto\n"
      "SphericalHarmonicL: [3, 255]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_8),
      Catch::Matchers::ContainsSubstring(
          "SphericalHarmonicL upper bound '255' "
          "cannot be larger than "
          "Spectral::limits::max_spherical_harmonic_mode"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Limits",
                  "[ParallelAlgorithms][Unit]") {
  test_equality();
  test_pup();
  test_option_parsing();
}
