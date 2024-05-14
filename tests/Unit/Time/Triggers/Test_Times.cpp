// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/Triggers/Times.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
  register_factory_classes_with_charm<Metavariables>();

  const auto check = [](const double time,
                        const std::vector<double>& trigger_times,
                        const bool expected) {
    CAPTURE(time);
    CAPTURE(trigger_times);
    const std::unique_ptr<Trigger> trigger = std::make_unique<Triggers::Times>(
        std::make_unique<TimeSequences::Specified<double>>(trigger_times));
    const auto sent_trigger = serialize_and_deserialize(trigger);

    const auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::Time>>(
        Metavariables{}, time);
    CHECK(trigger->is_triggered(box) == expected);
    CHECK(sent_trigger->is_triggered(box) == expected);
  };

  check(1.0, {}, false);
  check(1.0, {1.0}, true);
  check(1.0, {2.0}, false);
  check(1.0, {0.0, 1.0, 2.0}, true);
  check(1.0, {2.0, 1.0, 0.0}, true);

  TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
      "Times:\n"
      "  Specified:\n"
      "    Values: [2.0, 1.0, 3.0, 2.0]");
}
