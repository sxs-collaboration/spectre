// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DgSubcell/Tags/MeshForGhostData.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.Tags.MeshForGhostData",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::MeshForGhostData<1>>("MeshForGhostData");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::MeshForGhostData<2>>("MeshForGhostData");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::MeshForGhostData<3>>("MeshForGhostData");
}
