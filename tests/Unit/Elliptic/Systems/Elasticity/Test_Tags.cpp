// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Elasticity.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Elasticity::Tags::Displacement<1>>(
      "Displacement");
  TestHelpers::db::test_simple_tag<Elasticity::Tags::Strain<1>>("Strain");
  TestHelpers::db::test_simple_tag<Elasticity::Tags::Stress<1>>("Stress");
  TestHelpers::db::test_simple_tag<Elasticity::Tags::PotentialEnergy<1>>(
      "PotentialEnergy");
}
