// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <initializer_list>  // IWYU pragma: keep
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
#include "Time/Triggers/EveryNSlabs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <vector>

SPECTRE_TEST_CASE("Unit.Time.Triggers.EveryNSlabs", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::EveryNSlabs>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = test_factory_creation<TriggerType>(
      "  EveryNSlabs:\n"
      "    N: 3\n"
      "    Offset: 5");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(0., 1.);
  auto box = db::create<db::AddSimpleTags<Tags::TimeId>>(
      TimeId(true, 0, slab.start()));
  for (const bool expected :
       {false, false, false, false, false, true, false, false, true, false}) {
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<Tags::TimeId>(
        make_not_null(&box), [](const gsl::not_null<TimeId*> time_id) noexcept {
          *time_id = TimeId(true, time_id->slab_number(), time_id->time(), 1,
                            time_id->time());
        });
    CHECK_FALSE(sent_trigger->is_triggered(box));
    db::mutate<Tags::TimeId>(
        make_not_null(&box), [](const gsl::not_null<TimeId*> time_id) noexcept {
          *time_id = TimeId(true, time_id->slab_number() + 1, time_id->time());
        });
  }
}
