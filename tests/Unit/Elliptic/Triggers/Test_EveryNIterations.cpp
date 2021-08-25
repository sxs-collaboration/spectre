// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct OptionsGroup {};

struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation :
      tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Trigger,
        tmpl::list<elliptic::Triggers::EveryNIterations<OptionsGroup>>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Triggers.EveryNIterations",
                  "[Unit][Elliptic]") {
  Parallel::register_classes_with_charm<
      elliptic::Triggers::EveryNIterations<OptionsGroup>>();

  const auto trigger =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
          "EveryNIterations:\n"
          "  N: 3\n"
          "  Offset: 5");

  const auto sent_trigger = serialize_and_deserialize(trigger);

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Convergence::Tags::IterationId<OptionsGroup>>>(
      Metavariables{}, size_t{0});
  for (const bool expected :
       {false, false, false, false, false, true, false, false, true, false}) {
    CHECK(sent_trigger->is_triggered(box) == expected);
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id) noexcept {
          (*iteration_id)++;
        });
  }
}
