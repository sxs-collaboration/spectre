// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.Tag",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::fd::Tags::Reconstructor>("Reconstructor");
}
