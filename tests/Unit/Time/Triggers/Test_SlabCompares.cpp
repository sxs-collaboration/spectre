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
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/SlabCompares.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Time.Triggers.SlabCompares", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::SlabCompares>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = TestHelpers::test_factory_creation<TriggerType>(
      "SlabCompares:\n"
      "  Comparison: GreaterThan\n"
      "  Value: 3");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(0., 1.);
  {
    const auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(
        TimeStepId(true, 2, slab.start()));
    CHECK(not trigger->is_triggered(box));
    CHECK(not sent_trigger->is_triggered(box));
  }
  {
    const auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(
        TimeStepId(true, 3, slab.start()));
    CHECK(not trigger->is_triggered(box));
    CHECK(not sent_trigger->is_triggered(box));
  }
  {
    const auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(
        TimeStepId(true, 4, slab.start()));
    CHECK(trigger->is_triggered(box));
    CHECK(sent_trigger->is_triggered(box));
  }
}
