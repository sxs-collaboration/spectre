// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeId.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {
struct TimeTriggers {
  template <typename T>
  using type = Triggers::time_triggers<T>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Triggers.SpecifiedSlabs", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<Trigger<TimeTriggers>>();

  const auto trigger = test_factory_creation<Trigger<TimeTriggers>>(
      "  SpecifiedSlabs:\n"
      "    Slabs: [3, 6, 8]");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(0., 1.);
  auto box = db::create<db::AddTags<Tags::TimeId, Tags::TimeStep>,
                        db::AddComputeItemsTags<Tags::Time, Tags::TimeValue>>(
      TimeId{0, slab.start(), 0}, slab.duration());
  for (const bool expected :
       {false, false, false, true, false, false, true, false, true, false}) {
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<Tags::TimeId>(box,
                             [](TimeId& time_id) { ++time_id.slab_number; });
  }
}
