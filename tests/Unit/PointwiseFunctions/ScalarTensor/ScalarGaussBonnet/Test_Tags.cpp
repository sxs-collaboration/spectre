// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/RampUpFunction.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/Tags.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.Sgb.Tags",
                  "[Unit][PointwiseFunctions]") {
  TestHelpers::db::test_simple_tag<ScalarTensor::Tags::CouplingParameters>(
      "CouplingParameters");
  TestHelpers::test_option_tag<ScalarTensor::OptionTags::CouplingParameters>(
      "Linear: 2.0\n"
      "Quadratic: 0.1\n"
      "Quartic: 0.5\n");
}
