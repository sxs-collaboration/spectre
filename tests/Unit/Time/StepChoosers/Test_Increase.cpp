// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Increase", "[Unit][Time]") {
  using StepChooserType =
      StepChooser<tmpl::list<StepChoosers::Registrars::Increase>>;
  using Increase = StepChoosers::Increase<>;

  Parallel::register_derived_classes_with_charm<StepChooserType>();

  const Parallel::GlobalCache<Metavariables> cache{};
  auto box = db::create<db::AddSimpleTags<>>();
  const auto check =
      [&box, &cache](const double step, const double expected) noexcept {
    const Increase increase{5.};
    const std::unique_ptr<StepChooserType> increase_base =
        std::make_unique<Increase>(increase);

    CHECK(increase(step, cache) == std::make_pair(expected, true));
    CHECK(increase_base->desired_step(make_not_null(&box), step, cache) ==
          std::make_pair(expected, true));
    CHECK(increase_base->desired_slab(step, box, cache) == expected);
    CHECK(serialize_and_deserialize(increase)(step, cache) ==
          std::make_pair(expected, true));
    CHECK(serialize_and_deserialize(increase_base)
              ->desired_step(make_not_null(&box), step, cache) ==
          std::make_pair(expected, true));
  };
  check(0.25, 1.25);
  check(std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());

  TestHelpers::test_factory_creation<StepChooserType>(
      "Increase:\n"
      "  Factor: 5.0");
}
