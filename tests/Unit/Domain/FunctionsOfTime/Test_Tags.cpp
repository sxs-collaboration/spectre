// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {
SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.Tags", "[Domain][Unit]") {
  TestHelpers::db::test_base_tag<Tags::FunctionsOfTime>("FunctionsOfTime");
}
}  // namespace domain
