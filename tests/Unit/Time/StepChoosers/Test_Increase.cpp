// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Time.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Increase<StepChooserUse::LtsStep>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::list<StepChoosers::Increase<StepChooserUse::Slab>>>>;
  };
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Increase", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  const Parallel::GlobalCache<Metavariables> cache{};
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>>>(
      Metavariables{});
  const auto check =
      [&box, &cache](const double step, const double expected) noexcept {
    {
      const StepChoosers::Increase<StepChooserUse::LtsStep> increase{5.};
      const std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>
          increase_base =
              std::make_unique<StepChoosers::Increase<StepChooserUse::LtsStep>>(
                  increase);

      CHECK(increase(step, cache) == std::make_pair(expected, true));
      CHECK(serialize_and_deserialize(increase)(step, cache) ==
            std::make_pair(expected, true));
      CHECK(increase_base->desired_step(make_not_null(&box), step, cache) ==
            std::make_pair(expected, true));
      CHECK(serialize_and_deserialize(increase_base)
                ->desired_step(make_not_null(&box), step, cache) ==
            std::make_pair(expected, true));
    }
    {
      const StepChoosers::Increase<StepChooserUse::Slab> increase{5.};
      const std::unique_ptr<StepChooser<StepChooserUse::Slab>> increase_base =
          std::make_unique<StepChoosers::Increase<StepChooserUse::Slab>>(
              increase);

      CHECK(increase(step, cache) == std::make_pair(expected, true));
      CHECK(serialize_and_deserialize(increase)(step, cache) ==
            std::make_pair(expected, true));
      CHECK(increase_base->desired_slab(step, box, cache) == expected);
      CHECK(serialize_and_deserialize(increase_base)
                ->desired_slab(step, box, cache) == expected);
    }
  };
  check(0.25, 1.25);
  check(std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());

  TestHelpers::test_creation<
      std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>, Metavariables>(
      "Increase:\n"
      "  Factor: 5.0");
  TestHelpers::test_creation<std::unique_ptr<StepChooser<StepChooserUse::Slab>>,
                             Metavariables>(
      "Increase:\n"
      "  Factor: 5.0");
}
