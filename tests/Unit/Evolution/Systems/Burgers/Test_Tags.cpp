// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Tags", "[Unit][Burgers]") {
  TestHelpers::db::test_simple_tag<Burgers::Tags::U>("U");
}
