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

template <typename DataType, size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GammaHatMinusContractedConformalChristoffel<DataType, Dim,
                                                              Frame>>(
      "GammaHatMinusContractedConformalChristoffel");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::KMinus2ThetaC<DataType>>(
      "KMinus2ThetaC");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::KMinusK0Minus2ThetaC<DataType>>(
      "KMinusK0Minus2ThetaC");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ContractedFieldB<DataType>>(
      "ContractedFieldB");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetricTimesFieldB<DataType, Dim, Frame>>(
      "ConformalMetricTimesFieldB");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesRicciScalarPlus2DivergenceZ4Constraint<DataType>>(
      "LapseTimesRicciScalarPlus2DivergenceZ4Constraint");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetricTimesTraceATilde<DataType, Dim, Frame>>(
      "ConformalMetricTimesTraceATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesATilde<DataType, Dim, Frame>>("LapseTimesATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::FieldDUpTimesATilde<DataType, Dim, Frame>>(
      "FieldDUpTimesATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesDerivATilde<DataType, Dim, Frame>>(
      "LapseTimesDerivATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseConformalMetricTimesDerivATilde<DataType, Dim, Frame>>(
      "InverseConformalMetricTimesDerivATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ATildeMinusOneThirdConformalMetricTimesTraceATilde<
          DataType, Dim, Frame>>(
      "ATildeMinusOneThirdConformalMetricTimesTraceATilde");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesFieldA<DataType, Dim, Frame>>("LapseTimesFieldA");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ShiftTimesDerivGammaHat<DataType, Dim, Frame>>(
      "ShiftTimesDerivGammaHat");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseTauTimesConformalMetric<DataType, Dim, Frame>>(
      "InverseTauTimesConformalMetric");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::LapseTimesSlicingCondition<DataType>>(
      "LapseTimesSlicingCondition");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TempTags", "[Unit][Evolution]") {
  test_simple_tags<double, 1, ArbitraryFrame>();
  test_simple_tags<DataVector, 1, ArbitraryFrame>();
  test_simple_tags<double, 2, ArbitraryFrame>();
  test_simple_tags<DataVector, 2, ArbitraryFrame>();
  test_simple_tags<double, 3, ArbitraryFrame>();
  test_simple_tags<DataVector, 3, ArbitraryFrame>();
}
