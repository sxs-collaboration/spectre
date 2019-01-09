// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <pup.h>
// IWYU pragma: no_include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/Triggers/PastTime.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Time.Triggers.PastTime", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::PastTime>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = test_factory_creation<TriggerType>("  PastTime: -7.");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(-10., 10.);

  const auto check = [&sent_trigger](const Time& time, const TimeDelta& step,
                                     const bool expected) noexcept {
    const auto box =
        db::create<db::AddSimpleTags<Tags::TimeId, Tags::TimeStep>,
                   db::AddComputeTags<Tags::Time, Tags::TimeValue>>(
            TimeId(step.is_positive(), 0, time), step);
    CHECK(sent_trigger->is_triggered(box) == expected);
  };
  check(slab.start(), slab.duration(), false);
  check(slab.start(), -slab.duration(), true);
  check(slab.end(), slab.duration(), true);
  check(slab.end(), -slab.duration(), false);
}
