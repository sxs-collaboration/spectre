// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarTensor.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      ScalarTensor::Tags::TraceReversedStressEnergy>(
      "TraceReversedStressEnergy");
}
