// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<evolution::Tags::EventsAndDenseTriggers>(
      "EventsAndDenseTriggers");
}
