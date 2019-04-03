// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/Triggers/PastTime.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <vector>

SPECTRE_TEST_CASE("Unit.Time.Triggers.PastTime", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::PastTime>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = test_factory_creation<TriggerType>("  PastTime: -7.");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(-10., 10.);

  const auto check = [&sent_trigger](const Time& time,
                                     const bool time_runs_forward,
                                     const bool expected) noexcept {
    auto box = db::create<db::AddSimpleTags<Tags::TimeId>,
                          db::AddComputeTags<Tags::Time>>(
        TimeId(time_runs_forward, 0, time));
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<Tags::TimeId>(
        make_not_null(&box), [](const gsl::not_null<TimeId*> time_id) noexcept {
          *time_id =
              TimeId(time_id->time_runs_forward(), time_id->slab_number(),
                     time_id->time(), 1, time_id->time());
        });
    CHECK_FALSE(sent_trigger->is_triggered(box));
  };
  check(slab.start(), true, false);
  check(slab.start(), false, true);
  check(slab.end(), true, true);
  check(slab.end(), false, false);
}
