// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Time.Triggers.TimeCompares", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::TimeCompares>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = TestHelpers::test_creation<std::unique_ptr<TriggerType>>(
      "TimeCompares:\n"
      "  Comparison: GreaterThan\n"
      "  Value: 3.5");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  {
    const auto box = db::create<db::AddSimpleTags<Tags::Time>>(3.4);
    CHECK(not trigger->is_triggered(box));
    CHECK(not sent_trigger->is_triggered(box));
  }
  {
    const auto box = db::create<db::AddSimpleTags<Tags::Time>>(3.5);
    CHECK(not trigger->is_triggered(box));
    CHECK(not sent_trigger->is_triggered(box));
  }
  {
    const auto box = db::create<db::AddSimpleTags<Tags::Time>>(3.6);
    CHECK(trigger->is_triggered(box));
    CHECK(sent_trigger->is_triggered(box));
  }
}
