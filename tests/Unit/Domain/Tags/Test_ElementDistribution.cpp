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

SPECTRE_TEST_CASE("Unit.Domain.Tags.ElementDistribution", "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<domain::Tags::ElementDistribution>(
      "ElementDistribution");
  const auto make_option = [](const std::string& option_string)
      -> std::optional<domain::ElementWeight> {
    return TestHelpers::test_option_tag<
        domain::OptionTags::ElementDistribution>(option_string);
  };
  CHECK(make_option("Uniform") ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option("NumGridPoints") ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK(make_option("NumGridPointsAndGridSpacing") ==
        std::optional{domain::ElementWeight::NumGridPointsAndGridSpacing});
  CHECK(make_option("RoundRobin") == std::nullopt);
}
