// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Domain/Tags/ExternalBoundaryConditions.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Tags.ExternalBoundaryConditions",
                  "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<Tags::ExternalBoundaryConditions<3>>(
      "ExternalBoundaryConditions");
}

}  // namespace domain
