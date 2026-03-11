// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.BbhCompletionCriteria",
    "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks>(
      "MinCommonHorizonSuccessesBeforeChecks");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::MaxCommonHorizonSuccesses>(
      "MaxCommonHorizonSuccesses");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::GaugeConstraintLinfThreshold>(
      "GaugeConstraintLinfThreshold");
  TestHelpers::db::test_simple_tag<
      gh::bbh::Tags::ThreeIndexConstraintLinfThreshold>(
      "ThreeIndexConstraintLinfThreshold");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::CommonHorizonLMaxThreshold>(
      "CommonHorizonLMaxThreshold");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::ConstraintCheckVerbose>(
      "ConstraintCheckVerbose");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::GaugeConstraintExceeded>(
      "GaugeConstraintExceeded");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::ThreeIndexConstraintExceeded>(
      "ThreeIndexConstraintExceeded");
  TestHelpers::db::test_simple_tag<
      gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(
      "CommonHorizonLMaxBelowOrEqualThreshold");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::CommonHorizonSuccessCount>(
      "CommonHorizonSuccessCount");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::CompletionRequested>(
      "CompletionRequested");
  TestHelpers::db::test_simple_tag<gh::bbh::Tags::ElementCompletionRequested>(
      "ElementCompletionRequested");
}
