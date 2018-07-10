// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/ThermalNoise/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.ThermalNoise.Tags",
                  "[Unit][Elliptic]") {
  CHECK(ThermalNoise::Displacement<1>::name() == "Displacement");
  CHECK(ThermalNoise::Strain<1>::name() == "Strain");
}
