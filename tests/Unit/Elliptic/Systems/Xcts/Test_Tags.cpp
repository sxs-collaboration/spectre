// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

struct DataVector;
namespace Frame {
struct Inertial;
}  // namespace Frame

namespace Xcts {

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Tags::ConformalFactor<DataVector>>(
      "ConformalFactor");
  TestHelpers::db::test_simple_tag<Tags::LapseTimesConformalFactor<DataVector>>(
      "LapseTimesConformalFactor");
  TestHelpers::db::test_simple_tag<
      Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>("ShiftStrain");
  TestHelpers::db::test_prefix_tag<
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 2>>(
      "Conformal(EnergyDensity)");
  TestHelpers::db::test_simple_tag<
      Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
      "Conformal(SpatialMetric)");
  TestHelpers::db::test_simple_tag<
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>("ShiftBackground");
  TestHelpers::db::test_simple_tag<
      Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>("ShiftExcess");
  TestHelpers::db::test_simple_tag<
      Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>("ShiftStrain");
  TestHelpers::db::test_simple_tag<
      Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>(
      "LongitudinalShiftExcess");
  TestHelpers::db::test_simple_tag<
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>>(
      "LongitudinalShiftBackgroundMinusDtConformalMetric");
  TestHelpers::db::test_simple_tag<
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>>(
      "LongitudinalShiftMinusDtConformalMetricSquare");
  TestHelpers::db::test_simple_tag<
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataVector>>(
      "LongitudinalShiftMinusDtConformalMetricOverLapseSquare");
  TestHelpers::db::test_simple_tag<
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>(
      "ShiftDotDerivExtrinsicCurvatureTrace");
  TestHelpers::db::test_simple_tag<
      Tags::ConformalChristoffelFirstKind<DataVector, 3, Frame::Inertial>>(
      "ConformalChristoffelFirstKind");
  TestHelpers::db::test_simple_tag<
      Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>>(
      "ConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>>(
      "ConformalChristoffelContracted");
  TestHelpers::db::test_simple_tag<
      Tags::ConformalRicciTensor<DataVector, 3, Frame::Inertial>>(
      "ConformalRicciTensor");
  TestHelpers::db::test_simple_tag<Tags::ConformalRicciScalar<DataVector>>(
      "ConformalRicciScalar");
}

}  // namespace Xcts
