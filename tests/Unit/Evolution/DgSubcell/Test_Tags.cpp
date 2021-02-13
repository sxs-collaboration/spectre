// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tags",
                  "[Evolution][Unit]") {
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::ActiveGrid>(
      "ActiveGrid");
}
