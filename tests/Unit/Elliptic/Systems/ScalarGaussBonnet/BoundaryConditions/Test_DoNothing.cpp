// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/BoundaryConditions/DoNothing.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace sgb::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.ScalarGaussBonnet.BoundaryConditions.DoNothing",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, DoNothing>(
      "DoNothing");
  REQUIRE(dynamic_cast<const DoNothing*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const DoNothing&>(*created);
  // Test basic requirements for boundary conditions
  test_serialization(boundary_condition);
  test_copy_semantics(boundary_condition);
  auto move_boundary_condition = boundary_condition;
  test_move_semantics(std::move(move_boundary_condition), boundary_condition);
}
}  // namespace sgb::BoundaryConditions
