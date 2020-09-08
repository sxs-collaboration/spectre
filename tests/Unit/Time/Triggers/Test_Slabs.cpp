// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <initializer_list>
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
#include "Time/Triggers/Slabs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Time.Triggers.Slabs", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::Slabs>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();
  Parallel::register_derived_classes_with_charm<TimeSequence<std::uint64_t>>();

  const auto trigger = TestHelpers::test_factory_creation<TriggerType>(
      "Slabs:\n"
      "  Specified:\n"
      "    Values: [3, 6, 8]");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  const Slab slab(0., 1.);
  auto box = db::create<db::AddSimpleTags<Tags::TimeStepId>>(
      TimeStepId(true, 0, slab.start()));
  for (const bool expected :
       {false, false, false, true, false, false, true, false, true, false}) {
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
