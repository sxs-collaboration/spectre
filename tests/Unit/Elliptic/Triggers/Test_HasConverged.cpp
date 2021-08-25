// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Elliptic/Triggers/HasConverged.hpp"
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
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Trigger, tmpl::list<elliptic::Triggers::HasConverged<OptionsGroup>>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Triggers.HasConverged", "[Unit][Elliptic]") {
  Parallel::register_classes_with_charm<
      elliptic::Triggers::HasConverged<OptionsGroup>>();

  const auto created =
      TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
          "HasConverged");
  const auto serialized = serialize_and_deserialize(created);
  const auto& trigger = *serialized;

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Convergence::Tags::HasConverged<OptionsGroup>>>(
      Metavariables{}, Convergence::HasConverged{});
  CHECK_FALSE(trigger.is_triggered(box));
  db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
      make_not_null(&box), [](const gsl::not_null<Convergence::HasConverged*>
                                  has_converged) noexcept {
        *has_converged = Convergence::HasConverged{0, 0};
      });
  CHECK(trigger.is_triggered(box));
}
