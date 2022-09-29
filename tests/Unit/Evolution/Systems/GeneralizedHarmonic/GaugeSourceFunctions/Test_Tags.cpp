// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.Tags",
                  "[Evolution][Unit]") {
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::gauges::Tags::GaugeCondition>("GaugeCondition");
}
