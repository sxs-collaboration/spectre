// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.FilterOptions",
    "[Unit][Evolution]") {
  CHECK(not serialize_and_deserialize(
                TestHelpers::test_creation<
                    grmhd::GhValenciaDivClean::fd::FilterOptions>(
                    "SpacetimeDissipation: None"))
                .spacetime_dissipation.has_value());
  CHECK(serialize_and_deserialize(
            TestHelpers::test_creation<
                grmhd::GhValenciaDivClean::fd::FilterOptions>(
                "SpacetimeDissipation: 0.1"))
            .spacetime_dissipation.value() == 0.1);
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<grmhd::GhValenciaDivClean::fd::FilterOptions>(
          "SpacetimeDissipation: 0.0"),
      Catch::Matchers::Contains(
          "Spacetime dissipation must be between 0 and 1, but got"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<grmhd::GhValenciaDivClean::fd::FilterOptions>(
          "SpacetimeDissipation: 1.0"),
      Catch::Matchers::Contains(
          "Spacetime dissipation must be between 0 and 1, but got"));

  CHECK(serialize_and_deserialize(
            TestHelpers::test_option_tag<
                grmhd::GhValenciaDivClean::fd::OptionTags::FilterOptions>(
                "SpacetimeDissipation: 0.1"))
            .spacetime_dissipation.value() == 0.1);
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::fd::Tags::FilterOptions>("FilterOptions");
}
