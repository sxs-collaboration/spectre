// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Fd.Tag",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<ForceFree::fd::Tags::Reconstructor>(
      "Reconstructor");
}
