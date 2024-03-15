// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.EventsAndDenseTriggers.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<::Tags::EventsAndDenseTriggers>(
      "EventsAndDenseTriggers");
}
