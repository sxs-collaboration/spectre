// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/VariablesTag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Time/Tags/PreviousStepperError.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using DummyVariablesTag =
    Tags::Variables<tmpl::list<TestHelpers::Tags::Scalar<>>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags.PreviousStepperError", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<
      Tags::PreviousStepperError<DummyVariablesTag>>(
      "PreviousStepperError(Variables(Scalar))");
}
