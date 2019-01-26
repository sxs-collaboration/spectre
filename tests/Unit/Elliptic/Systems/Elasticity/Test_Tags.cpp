// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Elasticity/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Elasticity.Tags",
                  "[Unit][Elliptic]") {
  CHECK(Elasticity::Tags::Displacement<1>::name() == "Displacement");
  CHECK(Elasticity::Tags::Strain<1>::name() == "Strain");
  CHECK(Elasticity::Tags::Stress<1>::name() == "Stress");
}
