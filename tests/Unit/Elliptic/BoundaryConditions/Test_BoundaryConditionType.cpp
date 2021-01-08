// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

namespace elliptic {

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditionType", "[Unit][Elliptic]") {
  CHECK(get_output(BoundaryConditionType::Dirichlet) == "Dirichlet");
  CHECK(get_output(BoundaryConditionType::Neumann) == "Neumann");
  CHECK(TestHelpers::test_creation<BoundaryConditionType>("Dirichlet") ==
        BoundaryConditionType::Dirichlet);
  CHECK(TestHelpers::test_creation<BoundaryConditionType>("Neumann") ==
        BoundaryConditionType::Neumann);
}

// [[OutputRegex, Failed to convert "nil" to elliptic::BoundaryConditionType.]]
SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditionType.ParseError",
                  "[Unit][Elliptic]") {
  ERROR_TEST();
  TestHelpers::test_creation<elliptic::BoundaryConditionType>("nil");
}

}  // namespace elliptic
