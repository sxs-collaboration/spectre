// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/OptionTags/TimeStepper.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<TimeStepper, tmpl::list<TimeSteppers::AdamsBashforth>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Time.OptionTags.TimeStepper", "[Unit][Time]") {
  const auto stepper =
      TestHelpers::test_option_tag<OptionTags::TimeStepper<TimeStepper>,
                                   Metavariables>(
          "AdamsBashforth:\n"
          "  Order: 7\n");
  CHECK(stepper->order() == 7);
}
}  // namespace
