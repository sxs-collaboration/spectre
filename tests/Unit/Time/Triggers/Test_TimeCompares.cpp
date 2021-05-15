// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Trigger, tmpl::list<Triggers::TimeCompares>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Triggers.TimeCompares", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  const auto trigger =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
          "TimeCompares:\n"
          "  Comparison: GreaterThan\n"
          "  Value: 3.5");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  {
    const auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::Time>>(
        Metavariables{}, 3.4);
    CHECK(not trigger->is_triggered(box));
    CHECK(not sent_trigger->is_triggered(box));
  }
  {
    const auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::Time>>(
        Metavariables{}, 3.5);
    CHECK(not trigger->is_triggered(box));
    CHECK(not sent_trigger->is_triggered(box));
  }
  {
    const auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::Time>>(
        Metavariables{}, 3.6);
    CHECK(trigger->is_triggered(box));
    CHECK(sent_trigger->is_triggered(box));
  }
}
