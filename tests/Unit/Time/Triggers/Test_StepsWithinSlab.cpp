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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/Triggers/StepsWithinSlab.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<TimeSequence<std::uint64_t>,
                             TimeSequences::all_time_sequences<std::uint64_t>>,
                  tmpl::pair<Trigger, tmpl::list<Triggers::StepsWithinSlab>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Triggers.StepsWithSlab", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  const auto trigger =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
          "StepsWithinSlab:\n"
          "  Specified:\n"
          "    Values: [3, 6, 8]");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::StepNumberWithinSlab>>(Metavariables{},
                                                     uint64_t{0});
  for (const bool expected :
       {false, false, false, true, false, false, true, false, true, false}) {
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<Tags::StepNumberWithinSlab>(
        [](const gsl::not_null<uint64_t*> step_within_slab) {
          ++(*step_within_slab);
        },
        make_not_null(&box));
  }
}
