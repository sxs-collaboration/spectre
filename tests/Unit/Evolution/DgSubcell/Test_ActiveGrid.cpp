// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.ActiveGrid", "[Evolution][Unit]") {
  CHECK("Dg" == get_output(evolution::dg::subcell::ActiveGrid::Dg));
  CHECK("Subcell" == get_output(evolution::dg::subcell::ActiveGrid::Subcell));
  CHECK(TestHelpers::test_creation<evolution::dg::subcell::ActiveGrid>("Dg") ==
        evolution::dg::subcell::ActiveGrid::Dg);
  CHECK(TestHelpers::test_creation<evolution::dg::subcell::ActiveGrid>(
            "Subcell") == evolution::dg::subcell::ActiveGrid::Subcell);
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::ActiveGrid>(
      "ActiveGrid");
  CHECK(TestHelpers::test_option_tag<
            evolution::dg::subcell::OptionTags::ActiveGrid>("Dg") ==
        evolution::dg::subcell::ActiveGrid::Dg);
  CHECK(TestHelpers::test_option_tag<
            evolution::dg::subcell::OptionTags::ActiveGrid>("Subcell") ==
        evolution::dg::subcell::ActiveGrid::Subcell);
}
