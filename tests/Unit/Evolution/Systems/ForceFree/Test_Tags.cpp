// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Tags",
                  "[Unit][Evolution]") {
  // Primitive variable tags
  TestHelpers::db::test_simple_tag<ForceFree::Tags::ElectricField>(
      "ElectricField");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::MagneticField>(
      "MagneticField");
  TestHelpers::db::test_simple_tag<
      ForceFree::Tags::ElectricDivergenceCleaningField>(
      "ElectricDivergenceCleaningField");
  TestHelpers::db::test_simple_tag<
      ForceFree::Tags::MagneticDivergenceCleaningField>(
      "MagneticDivergenceCleaningField");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::ChargeDensity>(
      "ChargeDensity");

  // Evolved variable tags
  TestHelpers::db::test_simple_tag<ForceFree::Tags::TildeE>("TildeE");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::TildeB>("TildeB");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::TildePsi>("TildePsi");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::TildePhi>("TildePhi");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::TildeQ>("TildeQ");

  // etc.
  TestHelpers::db::test_simple_tag<ForceFree::Tags::ElectricCurrentDensity>(
      "ElectricCurrentDensity");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::KappaPsi>("KappaPsi");
  TestHelpers::db::test_simple_tag<ForceFree::Tags::KappaPhi>("KappaPhi");
}
