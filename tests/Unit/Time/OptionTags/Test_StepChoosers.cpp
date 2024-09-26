// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <typeinfo>

#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/OptionTags/StepChoosers.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::LimitIncrease>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Time.OptionTags.StepChoosers", "[Unit][Time]") {
  const auto choosers =
      TestHelpers::test_option_tag<OptionTags::StepChoosers, Metavariables>(
          "- LimitIncrease:\n"
          "    Factor: 3.0\n");
  CHECK(choosers.size() == 1);
  const auto& quiet_compiler = *choosers[0];
  CHECK(typeid(quiet_compiler) == typeid(StepChoosers::LimitIncrease));
}
}  // namespace
