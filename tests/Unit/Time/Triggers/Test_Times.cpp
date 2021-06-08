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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/Times.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<TimeSequence<double>,
                             TimeSequences::all_time_sequences<double>>,
                  tmpl::pair<Trigger, tmpl::list<Triggers::Times>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Triggers.Times", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  const auto check = [](const double time, const double slab_size,
                        const std::vector<double>& trigger_times,
                        const bool expected) noexcept {
    CAPTURE(time);
    CAPTURE(slab_size);
    CAPTURE(trigger_times);
    const auto slab = Slab::with_duration_from_start(0.8 * time, slab_size);
    // None of the arguments here should matter, except that they are
    // based on the correct slab.
    const TimeStepId time_id(false, 12, slab.start() + slab.duration() / 13);

    const std::unique_ptr<Trigger> trigger = std::make_unique<Triggers::Times>(
        std::make_unique<TimeSequences::Specified<double>>(trigger_times));
    const auto sent_trigger = serialize_and_deserialize(trigger);

    const auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          Tags::TimeStepId, Tags::Time>>(
        Metavariables{}, time_id, time);
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

  TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
      "Times:\n"
      "  Specified:\n"
      "    Values: [2.0, 1.0, 3.0, 2.0]");
}
