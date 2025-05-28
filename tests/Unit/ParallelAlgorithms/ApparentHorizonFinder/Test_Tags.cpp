// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Tags",
                  "[ApparentHorizonFinder][Unit]") {
  TestHelpers::db::test_simple_tag<ah::Tags::FastFlow>("FastFlow");
  TestHelpers::db::test_simple_tag<
      ah::Tags::PreviousIterationStrahlkorper<::Frame::Distorted>>(
      "PreviousIterationStrahlkorper");
  TestHelpers::db::test_simple_tag<ah::Tags::FailedInterpolationIterations>(
      "FailedInterpolationIterations");
  TestHelpers::db::test_base_tag<ah::Tags::ObserveCentersBase>(
      "ObserveCentersBase");
  TestHelpers::db::test_simple_tag<ah::Tags::ObserveCenters>("ObserveCenters");
}
