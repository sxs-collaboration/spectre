// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Tags/BlockNamesAndGroups.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Tags.BlockNamesAndGroups", "[Unit][Domain]") {
  constexpr size_t Dim = 1;
  TestHelpers::db::test_simple_tag<Tags::BlockNames<Dim>>("BlockNames");
  TestHelpers::db::test_simple_tag<Tags::BlockGroups<Dim>>("BlockGroups");
}

}  // namespace domain
