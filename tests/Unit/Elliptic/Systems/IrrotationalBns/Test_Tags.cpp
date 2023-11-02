// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/IrrotationalBns/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.IrrotationalBns.Tags",
                  "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<IrrotationalBns::Tags::VelocityPotential>(
      "VelocityPotential");
  TestHelpers::db::test_simple_tag<IrrotationalBns::Tags::AuxiliaryVelocity>(
      "AuxialliaryVelocity");
}
