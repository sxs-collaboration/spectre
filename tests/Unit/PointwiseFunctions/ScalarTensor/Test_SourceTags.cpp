// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/RampUpFunction.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarSource.hpp"
#include "PointwiseFunctions/ScalarTensor/SourceTags.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.SourceTags",
                  "[Unit][PointwiseFunctions]") {
  TestHelpers::db::test_simple_tag<ScalarTensor::Tags::RampUpParameters>(
      "RampUpParameters");
  TestHelpers::test_option_tag<ScalarTensor::OptionTags::RampUpStart>("0.0");
  TestHelpers::test_option_tag<ScalarTensor::OptionTags::RampUpDuration>(
      "100.0");
  CHECK_THROWS_WITH(
      ScalarTensor::Tags::RampUpParameters::create_from_options(0.0, 0.0),
      Catch::Matchers::ContainsSubstring(
          "Ramp up duration time must be greater than zero"));
}
