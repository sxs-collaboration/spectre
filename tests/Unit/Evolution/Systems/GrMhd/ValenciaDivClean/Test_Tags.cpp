// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeeds>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildeD>(
      "TildeD");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildeTau>(
      "TildeTau");
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::TildeS<Frame::Grid>>("Grid_TildeS");
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::TildeB<Frame::Grid>>("Grid_TildeB");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildePhi>(
      "TildePhi");

  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>(
      "ConstraintDampingParameter");

  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::VariablesNeededFixing>(
          "VariablesNeededFixing");
}
