// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>

#include "Domain/ElementDistribution.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Tags.ElementDistribution", "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<domain::Tags::ElementDistribution>(
      "ElementDistribution");
  CHECK(TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution>(
            "Uniform") == std::optional{domain::ElementWeight::Uniform});
  CHECK(TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution>(
            "NumGridPoints") ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK(TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution>(
            "NumGridPointsAndGridSpacing") ==
        std::optional{domain::ElementWeight::NumGridPointsAndGridSpacing});
  CHECK(TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution>(
            "RoundRobin") == std::nullopt);
}
