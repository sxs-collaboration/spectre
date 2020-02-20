// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct DummyType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.EventsAndTriggers.Tags",
                  "[Unit][ParallelAlgorithms]") {
  TestHelpers::db::test_base_tag<Tags::EventsAndTriggersBase>(
      "EventsAndTriggersBase");
  TestHelpers::db::test_simple_tag<
      Tags::EventsAndTriggers<DummyType, DummyType>>("EventsAndTriggers");
}
