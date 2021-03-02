// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.ActiveGrid", "[Evolution][Unit]") {
  CHECK("Dg" == get_output(evolution::dg::subcell::ActiveGrid::Dg));
  CHECK("Subcell" == get_output(evolution::dg::subcell::ActiveGrid::Subcell));
}
