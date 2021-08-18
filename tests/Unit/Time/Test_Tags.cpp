// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct DummyType {};
struct DummyTag : db::SimpleTag {
  using type = DummyType;
};
using DummyVariablesTag =
    Tags::Variables<tmpl::list<TestHelpers::Tags::Scalar<>>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags", "[Unit][Time]") {
  TestHelpers::db::test_base_tag<Tags::HistoryEvolvedVariables<>>(
      "HistoryEvolvedVariables");
  TestHelpers::db::test_base_tag<Tags::TimeStepper<>>("TimeStepper");
  TestHelpers::db::test_compute_tag<Tags::SubstepTimeCompute>("SubstepTime");
  TestHelpers::db::test_prefix_tag<SelfStart::Tags::InitialValue<DummyTag>>(
      "InitialValue(DummyTag)");
  TestHelpers::db::test_simple_tag<Tags::TimeStepId>("TimeStepId");
  TestHelpers::db::test_simple_tag<Tags::TimeStep>("TimeStep");
  TestHelpers::db::test_simple_tag<Tags::SubstepTime>("SubstepTime");
  TestHelpers::db::test_simple_tag<Tags::Time>("Time");
  TestHelpers::db::test_simple_tag<
      Tags::HistoryEvolvedVariables<DummyVariablesTag>>(
      "HistoryEvolvedVariables");
  TestHelpers::db::test_simple_tag<
      Tags::BoundaryHistory<DummyType, DummyType, DummyType>>(
      "BoundaryHistory");
  TestHelpers::db::test_simple_tag<Tags::TimeStepper<DummyType>>("TimeStepper");
  TestHelpers::db::test_simple_tag<Tags::StepChoosers>("StepChoosers");
  TestHelpers::db::test_simple_tag<Tags::StepController>("StepController");
  TestHelpers::db::test_simple_tag<Tags::TimeAndPrevious>("TimeAndPrevious");
  TestHelpers::db::test_compute_tag<Tags::TimeAndPreviousCompute>(
      "TimeAndPrevious");
  TestHelpers::db::test_simple_tag<Tags::StepperError<DummyVariablesTag>>(
      "StepperError(Variables(Scalar))");
  TestHelpers::db::test_simple_tag<Tags::StepperErrorUpdated>(
      "StepperErrorUpdated");
}
