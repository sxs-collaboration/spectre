// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Fd.Tag",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<Burgers::fd::Tags::Reconstructor>(
      "Reconstructor");
}
