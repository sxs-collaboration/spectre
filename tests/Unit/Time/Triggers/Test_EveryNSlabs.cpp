// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/EveryNSlabs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <vector>

SPECTRE_TEST_CASE("Unit.Time.Triggers.EveryNSlabs", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::EveryNSlabs>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = TestHelpers::test_factory_creation<TriggerType>(
      "EveryNSlabs:\n"
      "  N: 3\n"
      "  Offset: 5");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(0., 1.);
  auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(
      TimeStepId(true, 0, slab.start()));
  for (const bool expected :
       {false, false, false, false, false, true, false, false, true, false}) {
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<Tags::TimeStepId>(make_not_null(&box), [
    ](const gsl::not_null<TimeStepId*> time_id) noexcept {
      *time_id = TimeStepId(true, time_id->slab_number(), time_id->step_time(),
                            1, time_id->step_time());
    });
    CHECK_FALSE(sent_trigger->is_triggered(box));
    db::mutate<Tags::TimeStepId>(make_not_null(&box), [
    ](const gsl::not_null<TimeStepId*> time_id) noexcept {
      *time_id =
          TimeStepId(true, time_id->slab_number() + 1, time_id->step_time());
    });
  }
}
