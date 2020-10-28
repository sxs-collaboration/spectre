// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/BoundaryConditions/Type.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.BoundaryConditions.Type",
                  "[Unit][Evolution]") {
  CHECK(get_output(evolution::BoundaryConditions::Type::Ghost) == "Ghost");
  CHECK(get_output(evolution::BoundaryConditions::Type::TimeDerivative) ==
        "TimeDerivative");
  CHECK(
      get_output(evolution::BoundaryConditions::Type::GhostAndTimeDerivative) ==
      "GhostAndTimeDerivative");
  CHECK(get_output(evolution::BoundaryConditions::Type::Outflow) == "Outflow");
}
