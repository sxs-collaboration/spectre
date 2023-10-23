// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Tags/NeighborMesh.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.Tags.NeighborMesh", "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<domain::Tags::NeighborMesh<1>>(
      "NeighborMesh");
  TestHelpers::db::test_simple_tag<domain::Tags::NeighborMesh<2>>(
      "NeighborMesh");
  TestHelpers::db::test_simple_tag<domain::Tags::NeighborMesh<3>>(
      "NeighborMesh");
}
