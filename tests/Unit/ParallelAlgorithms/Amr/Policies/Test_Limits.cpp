// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"

namespace {
void test_equality() {
  INFO("Equality");
  CHECK(amr::Limits{0, 3, 1, 5} == amr::Limits{0, 3, 1, 5});
  CHECK(amr::Limits{0, 3, 1, 5} != amr::Limits{1, 3, 1, 5});
  CHECK(amr::Limits{0, 3, 1, 5} != amr::Limits{0, 4, 1, 5});
  CHECK(amr::Limits{0, 3, 1, 5} != amr::Limits{0, 3, 2, 5});
  CHECK(amr::Limits{0, 3, 1, 5} != amr::Limits{0, 3, 1, 6});
}

void test_pup() {
  INFO("Serialization");
  test_serialization(amr::Limits{0, 3, 1, 5});
}

void test_option_parsing() {
  INFO("Option Parsing creation");
  {
    const std::string creation_string_1 =
        "RefinementLevel: [0, 3]\n"
        "NumGridPoints: [1, 5]\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_1);
    CHECK(limits == amr::Limits{0, 3, 1, 5});
  }

  {
    const std::string creation_string_2 =
        "RefinementLevel: Auto\n"
        "NumGridPoints: [1, 5]\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_2);
    CHECK(limits == amr::Limits{0, ElementId<1>::max_refinement_level, 1, 5});
  }

  {
    const std::string creation_string_3 =
        "RefinementLevel: [0, 3]\n"
        "NumGridPoints: Auto\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_3);
    CHECK(limits ==
          amr::Limits{
              0, 3, 1,
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre>});
  }

  {
    const std::string creation_string_4 =
        "RefinementLevel: Auto\n"
        "NumGridPoints: Auto\n"
        "ErrorBeyondLimits: False\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_4);
    CHECK(limits == amr::Limits{});
  }

  {
    const std::string creation_string_5 =
        "RefinementLevel: [0, 3]\n"
        "NumGridPoints: [1, 5]\n"
        "ErrorBeyondLimits: True\n";
    const auto limits =
        TestHelpers::test_creation<amr::Limits>(creation_string_5);
    CHECK(limits == amr::Limits{{{0, 3}}, {{1, 5}}, true});
  }

  const std::string bad_creation_string_1 =
      "RefinementLevel: [255, 3]\n"
      "NumGridPoints: [1, 5]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_1),
      Catch::Matchers::ContainsSubstring(
          "RefinementLevel lower bound '255' cannot be larger than '"));

  const std::string bad_creation_string_2 =
      "RefinementLevel: [1, 255]\n"
      "NumGridPoints: [1, 5]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_2),
      Catch::Matchers::ContainsSubstring(
          "RefinementLevel upper bound '255' cannot be larger than '"));

  const std::string bad_creation_string_3 =
      "RefinementLevel: [0, 3]\n"
      "NumGridPoints: [0, 5]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_3),
      Catch::Matchers::ContainsSubstring(
          "NumGridPoints lower bound '0' cannot be smaller than '1'."));

  const std::string bad_creation_string_4 =
      "RefinementLevel: [1, 3]\n"
      "NumGridPoints: [255, 5]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_4),
      Catch::Matchers::ContainsSubstring(
          "NumGridPoints lower bound '255' cannot be larger than '"));

  const std::string bad_creation_string_5 =
      "RefinementLevel: [1, 3]\n"
      "NumGridPoints: [1, 0]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_5),
      Catch::Matchers::ContainsSubstring(
          "NumGridPoints upper bound '0' cannot be smaller than '1'."));

  const std::string bad_creation_string_6 =
      "RefinementLevel: [1, 3]\n"
      "NumGridPoints: [1, 255]\n"
      "ErrorBeyondLimits: False\n";
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<amr::Limits>(bad_creation_string_6),
      Catch::Matchers::ContainsSubstring(
          "NumGridPoints upper bound '255' cannot be larger than '"));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Limits",
                  "[ParallelAlgorithms][Unit]") {
  test_equality();
  test_pup();
  test_option_parsing();
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        amr::Limits limits{2, 1, 1, 5};
      }()),
      Catch::Matchers::ContainsSubstring(
          "The minimum refinement level '2' cannot be larger than "
          "the maximum refinement level '1'"));
  CHECK_THROWS_WITH(([]() {
                      amr::Limits limits{0, 3, 6, 4};
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "The minimum resolution '6' cannot be larger than the "
                        "maximum resolution '4'"));
#endif  // SPECTRE_DEBUG
}
