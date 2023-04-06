// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/StepChoosers/Constant.hpp"
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
                   tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::list<StepChoosers::Constant<StepChooserUse::Slab>>>>;
  };
  using component_list = tmpl::list<>;
};

template <typename Use>
void test_use() {
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>>>(
      Metavariables{});

  const StepChoosers::Constant<Use> constant{5.4};
  const std::unique_ptr<StepChooser<Use>> constant_base =
      std::make_unique<StepChoosers::Constant<Use>>(constant);

  const double current_step = std::numeric_limits<double>::infinity();
  CHECK(constant(current_step) == std::make_pair(5.4, true));
  CHECK(serialize_and_deserialize(constant)(current_step) ==
        std::make_pair(5.4, true));
  CHECK(constant_base->desired_step(current_step, box) ==
        std::make_pair(5.4, true));
  CHECK(serialize_and_deserialize(constant_base)
            ->desired_step(current_step, box) == std::make_pair(5.4, true));

  TestHelpers::test_creation<std::unique_ptr<StepChooser<Use>>, Metavariables>(
      "Constant: 5.4");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Constant", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  test_use<StepChooserUse::LtsStep>();
  test_use<StepChooserUse::Slab>();

  CHECK(not StepChoosers::Constant<StepChooserUse::Slab>{}.uses_local_data());
}

// [[OutputRegex, Requested step magnitude should be positive]]
SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Constant.bad_create",
                  "[Unit][Time]") {
  ERROR_TEST();
  TestHelpers::test_creation<std::unique_ptr<StepChooser<StepChooserUse::Slab>>,
                             Metavariables>("Constant: -5.4");
}
