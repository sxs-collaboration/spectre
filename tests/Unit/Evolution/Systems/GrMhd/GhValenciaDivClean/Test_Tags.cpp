// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GhValenciaDivClean.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::Tags::TraceReversedStressEnergy>(
      "TraceReversedStressEnergy");
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::Tags::StressEnergy>("StressEnergy");
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::Tags::ComovingMagneticField>(
      "ComovingMagneticField");
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::Tags::ComovingMagneticFieldOneForm>(
          "ComovingMagneticFieldOneForm");
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::Tags::FourVelocity>(
          "FourVelocity");
  TestHelpers::db::test_simple_tag<
      grmhd::GhValenciaDivClean::Tags::FourVelocityOneForm>(
          "FourVelocityOneForm");
}
