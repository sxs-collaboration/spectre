// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Time/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct DummyType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::TimeStepId>("TimeStepId");
  TestHelpers::db::test_simple_tag<Tags::TimeStep>("TimeStep");
  TestHelpers::db::test_simple_tag<Tags::SubstepTime>("SubstepTime");
  TestHelpers::db::test_simple_tag<Tags::Time>("Time");
  TestHelpers::db::test_simple_tag<Tags::HistoryEvolvedVariables<DummyType>>(
      "HistoryEvolvedVariables");
  TestHelpers::db::test_simple_tag<
      Tags::BoundaryHistory<DummyType, DummyType, DummyType>>(
      "BoundaryHistory");
  TestHelpers::db::test_simple_tag<Tags::TimeStepper<DummyType>>("TimeStepper");
  TestHelpers::db::test_simple_tag<Tags::StepChoosers<DummyType>>(
      "StepChoosers");
  TestHelpers::db::test_simple_tag<Tags::StepController>("StepController");
}
