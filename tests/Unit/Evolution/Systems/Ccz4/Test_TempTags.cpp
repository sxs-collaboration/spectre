// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame, typename DataType>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GammaHatMinusContractedConformalChristoffel<Dim, Frame,
                                                              DataType>>(
      "GammaHatMinusContractedConformalChristoffel");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::KMinus2ThetaC<DataType>>(
      "KMinus2ThetaC");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::KMinusK0Minus2ThetaC<DataType>>(
      "KMinusK0Minus2ThetaC");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ContractedFieldB<DataType>>(
      "ContractedFieldB");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetricTimesFieldB<Dim, Frame, DataType>>(
      "ConformalMetricTimesFieldB");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesRicciScalarPlus2DivergenceZ4Constraint<DataType>>(
      "LapseTimesRicciScalarPlus2DivergenceZ4Constraint");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetricTimesTraceATilde<Dim, Frame, DataType>>(
      "ConformalMetricTimesTraceATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesATilde<Dim, Frame, DataType>>("LapseTimesATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::FieldDUpTimesATilde<Dim, Frame, DataType>>(
      "FieldDUpTimesATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesDerivATilde<Dim, Frame, DataType>>(
      "LapseTimesDerivATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseConformalMetricTimesDerivATilde<Dim, Frame, DataType>>(
      "InverseConformalMetricTimesDerivATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ATildeMinusOneThirdConformalMetricTimesTraceATilde<Dim, Frame,
                                                                     DataType>>(
      "ATildeMinusOneThirdConformalMetricTimesTraceATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesFieldA<Dim, Frame, DataType>>("LapseTimesFieldA");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ShiftTimesDerivGammaHat<Dim, Frame, DataType>>(
      "ShiftTimesDerivGammaHat");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseTauTimesConformalMetric<Dim, Frame, DataType>>(
      "InverseTauTimesConformalMetric");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesSlicingCondition<DataType>>(
      "LapseTimesSlicingCondition");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TempTags", "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame, double>();
  test_simple_tags<1, ArbitraryFrame, DataVector>();
  test_simple_tags<2, ArbitraryFrame, double>();
  test_simple_tags<2, ArbitraryFrame, DataVector>();
  test_simple_tags<3, ArbitraryFrame, double>();
  test_simple_tags<3, ArbitraryFrame, DataVector>();
}
