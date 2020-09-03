// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

struct DataVector;
namespace Frame {
struct Inertial;
}  // namespace Frame

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Xcts::Tags::ConformalFactor<DataVector>>(
      "ConformalFactor");
  TestHelpers::db::test_simple_tag<
      Xcts::Tags::LapseTimesConformalFactor<DataVector>>(
      "LapseTimesConformalFactor");
  TestHelpers::db::test_simple_tag<
      Xcts::Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>("ShiftStrain");
  TestHelpers::db::test_prefix_tag<
      Xcts::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 2>>(
      "Conformal(EnergyDensity)");
  TestHelpers::db::test_simple_tag<
      Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
      "Conformal(SpatialMetric)");
}
