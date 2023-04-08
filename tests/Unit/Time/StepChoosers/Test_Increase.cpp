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
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
  register_factory_classes_with_charm<Metavariables>();

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>>>(
      Metavariables{});
  const auto check = [&box](auto use, const double step,
                            const double expected) {
    using Use = tmpl::type_from<decltype(use)>;
    const StepChoosers::Increase<Use> increase{5.};
    const std::unique_ptr<StepChooser<Use>> increase_base =
        std::make_unique<StepChoosers::Increase<Use>>(increase);

    CHECK(increase(step) == std::make_pair(expected, true));
    CHECK(serialize_and_deserialize(increase)(step) ==
          std::make_pair(expected, true));
    CHECK(increase_base->desired_step(step, box) ==
          std::make_pair(expected, true));
    CHECK(serialize_and_deserialize(increase_base)->desired_step(step, box) ==
          std::make_pair(expected, true));
  };
  check(tmpl::type_<StepChooserUse::LtsStep>{}, 0.25, 1.25);
  check(tmpl::type_<StepChooserUse::Slab>{}, 0.25, 1.25);
  check(tmpl::type_<StepChooserUse::LtsStep>{},
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
  check(tmpl::type_<StepChooserUse::Slab>{},
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());

  TestHelpers::test_creation<
      std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>, Metavariables>(
      "Increase:\n"
      "  Factor: 5.0");
  TestHelpers::test_creation<std::unique_ptr<StepChooser<StepChooserUse::Slab>>,
                             Metavariables>(
      "Increase:\n"
      "  Factor: 5.0");

  CHECK(not StepChoosers::Increase<StepChooserUse::Slab>{}.uses_local_data());
}
