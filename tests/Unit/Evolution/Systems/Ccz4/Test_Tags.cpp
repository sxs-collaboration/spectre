// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <typename DataType, size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ConformalFactor<DataType>>(
      "ConformalFactor");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalFactorSquared<DataType>>("ConformalFactorSquared");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetric<DataType, Dim, Frame>>(
      "Conformal(SpatialMetric)");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseConformalMetric<DataType, Dim, Frame>>(
      "Conformal(InverseSpatialMetric)");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ATilde<DataType, Dim, Frame>>(
      "ATilde");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::TraceATilde<DataType>>(
      "TraceATilde");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::LogLapse<DataType>>("LogLapse");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldA<DataType, Dim, Frame>>(
      "FieldA");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldB<DataType, Dim, Frame>>(
      "FieldB");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldD<DataType, Dim, Frame>>(
      "FieldD");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::LogConformalFactor<DataType>>(
      "LogConformalFactor");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldP<DataType, Dim, Frame>>(
      "FieldP");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldDUp<DataType, Dim, Frame>>(
      "FieldDUp");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame>>(
      "ConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::DerivConformalChristoffelSecondKind<DataType, Dim, Frame>>(
      "DerivConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ChristoffelSecondKind<DataType, Dim, Frame>>(
      "ChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Ricci<DataType, Dim, Frame>>(
      "Ricci");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GradGradLapse<DataType, Dim, Frame>>("GradGradLapse");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::DivergenceLapse<DataType>>(
      "DivergenceLapse");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ContractedConformalChristoffelSecondKind<DataType, Dim,
                                                           Frame>>(
      "ContractedConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::DerivContractedConformalChristoffelSecondKind<DataType, Dim,
                                                                Frame>>(
      "DerivContractedConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::GammaHat<DataType, Dim, Frame>>(
      "GammaHat");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::SpatialZ4Constraint<DataType, Dim, Frame>>(
      "SpatialZ4Constraint");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::SpatialZ4ConstraintUp<DataType, Dim, Frame>>(
      "SpatialZ4ConstraintUp");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GradSpatialZ4Constraint<DataType, Dim, Frame>>(
      "GradSpatialZ4Constraint");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::RicciScalarPlusDivergenceZ4Constraint<DataType>>(
      "RicciScalarPlusDivergenceZ4Constraint");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Tags", "[Unit][Evolution]") {
  test_simple_tags<double, 1, ArbitraryFrame>();
  test_simple_tags<DataVector, 1, ArbitraryFrame>();
  test_simple_tags<double, 2, ArbitraryFrame>();
  test_simple_tags<DataVector, 2, ArbitraryFrame>();
  test_simple_tags<double, 3, ArbitraryFrame>();
  test_simple_tags<DataVector, 3, ArbitraryFrame>();
}
