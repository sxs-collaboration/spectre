// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>

#include "Domain/ElementDistribution.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/Parallelization.hpp"

namespace {
template <bool UseLTS>
struct TestMetavars {
  static constexpr bool local_time_stepping = UseLTS;
};

template <bool UseLTS>
std::optional<domain::ElementWeight> make_option(
    const std::string& option_string) {
  return TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution,
                                      TestMetavars<UseLTS>>(option_string);
}

std::optional<domain::ElementWeight> make_option_without_lts_metavars(
    const std::string& option_string) {
  return TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution>(
      option_string);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Tags.ElementDistribution", "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<domain::Tags::ElementDistribution>(
      "ElementDistribution");
  CHECK(make_option<true>("Uniform") ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option<true>("NumGridPoints") ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK(make_option<true>("NumGridPointsAndGridSpacing") ==
        std::optional{domain::ElementWeight::NumGridPointsAndGridSpacing});
  CHECK(make_option<true>("RoundRobin") == std::nullopt);

  CHECK(make_option<false>("Uniform") ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option<false>("NumGridPoints") ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK_THROWS_WITH(make_option<false>("NumGridPointsAndGridSpacing"),
                    Catch::Matchers::ContainsSubstring(
                        "When not using local time stepping") and
                        Catch::Matchers::ContainsSubstring(
                            "Please choose another element distribution."));
  CHECK(make_option<false>("RoundRobin") == std::nullopt);

  CHECK(make_option_without_lts_metavars("Uniform") ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option_without_lts_metavars("NumGridPoints") ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK_THROWS_WITH(
      make_option_without_lts_metavars("NumGridPointsAndGridSpacing"),
      Catch::Matchers::ContainsSubstring(
          "When not using local time stepping") and
          Catch::Matchers::ContainsSubstring(
              "Please choose another element distribution."));
  CHECK(make_option_without_lts_metavars("RoundRobin") == std::nullopt);
}
