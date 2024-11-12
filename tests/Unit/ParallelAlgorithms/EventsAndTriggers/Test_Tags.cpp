// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.EventsAndTriggers.Tags",
                  "[Unit][ParallelAlgorithms]") {
  TestHelpers::db::test_simple_tag<
      Tags::EventsAndTriggers<Triggers::WhenToCheck::AtIterations>>(
      "EventsAndTriggersAtIterations");
  TestHelpers::db::test_simple_tag<
      Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
      "EventsAndTriggersAtSlabs");
  TestHelpers::db::test_simple_tag<
      Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSteps>>(
      "EventsAndTriggersAtSteps");
  TestHelpers::db::test_simple_tag<Tags::EventsRunAtCleanup>(
      "EventsRunAtCleanup");
  TestHelpers::db::test_simple_tag<Tags::EventsRunAtCleanupObservationValue>(
      "EventsRunAtCleanupObservationValue");
}
