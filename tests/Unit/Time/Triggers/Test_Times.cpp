// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/Times.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Time.Triggers.Times", "[Unit][Time]") {
  using TriggerType = Trigger<tmpl::list<Triggers::Registrars::Times>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();
  Parallel::register_derived_classes_with_charm<TimeSequence<double>>();

  const auto check = [](const double time, const double slab_size,
                        const std::vector<double>& trigger_times,
                        const bool expected) noexcept {
    CAPTURE_PRECISE(time);
    CAPTURE_PRECISE(slab_size);
    CAPTURE_PRECISE(trigger_times);
    const auto slab = Slab::with_duration_from_start(0.8 * time, slab_size);
    // None of the arguments here should matter, except that they are
    // based on the correct slab.
    const TimeStepId time_id(false, 12, slab.start() + slab.duration() / 13);

    const std::unique_ptr<TriggerType> trigger =
        std::make_unique<Triggers::Times<>>(
            std::make_unique<TimeSequences::Specified<double>>(trigger_times));
    const auto sent_trigger = serialize_and_deserialize(trigger);

    const auto box =
        db::create<db::AddSimpleTags<Tags::TimeStepId, Tags::Time>>(time_id,
                                                                    time);
    CHECK(trigger->is_triggered(box) == expected);
    CHECK(sent_trigger->is_triggered(box) == expected);
  };

  static constexpr double infinity = std::numeric_limits<double>::infinity();

  check(1.0, 1.0, {}, false);
  check(1.0, 1.0, {1.0}, true);
  check(1.0, 1.0, {2.0}, false);
  check(1.0, 1.0, {0.0, 1.0, 2.0}, true);
  check(1.0, 1.0, {2.0, 1.0, 0.0}, true);
  check(std::nextafter(1.0, +infinity), 1.0, {0.0, 1.0, 2.0}, true);
  check(std::nextafter(1.0, -infinity), 1.0, {0.0, 1.0, 2.0}, true);

  check(1.0e5, 1.0, {1.0e5}, true);
  check(std::nextafter(1.0e5, +infinity), 1.0, {1.0e5}, true);
  check(std::nextafter(1.0e5, -infinity), 1.0, {1.0e5}, true);

  const double inaccurate_1 = 1.0 + 1.0e5 - std::nextafter(1.0e5, -infinity);
  check(inaccurate_1, 1.0, {1.0}, false);
  check(inaccurate_1, 1.0e5, {1.0}, true);

  TestHelpers::test_factory_creation<TriggerType>(
      "Times:\n"
      "  Specified:\n"
      "    Values: [2.0, 1.0, 3.0, 2.0]");
}
