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
#include "Time/StepChoosers/Maximum.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Maximum<StepChooserUse::LtsStep>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::list<StepChoosers::Maximum<StepChooserUse::Slab>>>>;
  };
  using component_list = tmpl::list<>;
};

template <typename Use>
void test_use() {
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>>>(
      Metavariables{});

  // Sign of argument should be ignored.
  const StepChoosers::Maximum<Use> maximum{-5.4};
  const std::unique_ptr<StepChooser<Use>> maximum_base =
      std::make_unique<StepChoosers::Maximum<Use>>(maximum);

  const double current_step = std::numeric_limits<double>::infinity();
  const std::pair<TimeStepRequest, bool> expected{{.size = 5.4}, true};
  CHECK(maximum(current_step) == expected);
  CHECK(serialize_and_deserialize(maximum)(current_step) == expected);
  CHECK(maximum_base->desired_step(current_step, box) == expected);
  CHECK(serialize_and_deserialize(maximum_base)
            ->desired_step(current_step, box) == expected);

  TestHelpers::test_creation<std::unique_ptr<StepChooser<Use>>, Metavariables>(
      "Maximum: 5.4");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Maximum", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  test_use<StepChooserUse::LtsStep>();
  test_use<StepChooserUse::Slab>();

  CHECK(not StepChoosers::Maximum<StepChooserUse::Slab>{}.uses_local_data());
}
