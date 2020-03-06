// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct IterationIdTag : db::SimpleTag {
  using type = size_t;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Triggers.EveryNIterations",
                  "[Unit][Elliptic]") {
  using TriggerType = Trigger<tmpl::list<
      elliptic::Triggers::Registrars::EveryNIterations<IterationIdTag>>>;
  Parallel::register_derived_classes_with_charm<TriggerType>();

  const auto trigger = TestHelpers::test_factory_creation<TriggerType>(
      "EveryNIterations:\n"
      "  N: 3\n"
      "  Offset: 5");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  auto box = db::create<db::AddSimpleTags<IterationIdTag>>(size_t{0});
  for (const bool expected :
       {false, false, false, false, false, true, false, false, true, false}) {
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<IterationIdTag>(
        make_not_null(&box), [](const gsl::not_null<size_t*>
                                    iteration_id) noexcept {
          (*iteration_id)++;
        });
  }
}
