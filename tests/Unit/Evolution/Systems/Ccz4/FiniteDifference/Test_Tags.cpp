// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <typename DataType, size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::Reconstructor>(
      "Reconstructor");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::EvolveLapseAndShift>(
      "EvolveLapseAndShift");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::ConstrainedEvolution>(
      "ConstrainedEvolution");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::KreissOligerEpsilon>(
      "KreissOligerEpsilon");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::CPhi<DataType>>("CPhi");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::CGamma<DataType>>("CGamma");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::CAlpha<DataType>>("CAlpha");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::CK<DataType>>("CK");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::CTheta<DataType>>("CTheta");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::CBeta<DataType>>("CBeta");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::CharacteristicSpeeds<DataType>>("CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UTensorPlus<DataType, Dim, Frame>>("UTensorPlus");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UTensorMinus<DataType, Dim, Frame>>("UTensorMinus");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UVector1Zero<DataType, Dim, Frame>>("UVector1Zero");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UVector2Plus<DataType, Dim, Frame>>("UVector2Plus");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UVector2Minus<DataType, Dim, Frame>>("UVector2Minus");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UVector3Plus<DataType, Dim, Frame>>("UVector3Plus");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::UVector3Minus<DataType, Dim, Frame>>("UVector3Minus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar1Zero<DataType>>(
      "UScalar1Zero");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar2Plus<DataType>>(
      "UScalar2Plus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar2Minus<DataType>>(
      "UScalar2Minus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar3Plus<DataType>>(
      "UScalar3Plus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar3Minus<DataType>>(
      "UScalar3Minus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar4Plus<DataType>>(
      "UScalar4Plus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar4Minus<DataType>>(
      "UScalar4Minus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar5Plus<DataType>>(
      "UScalar5Plus");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::UScalar5Minus<DataType>>(
      "UScalar5Minus");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::CharacteristicFields<DataType, Dim, Frame>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::DnConformalMetric<DataType, Dim, Frame>>(
      "DnConformalMetric");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::DnLapse<DataType>>(
      "DnLapse");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::DnShift<DataType, Dim, Frame>>("DnShift");
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::DnConformalFactor<DataType>>(
      "DnConformalFactor");
  TestHelpers::db::test_simple_tag<
      Ccz4::fd::Tags::EvolvedSpaceFromCharacteristicFields<DataType, Dim,
                                                           Frame>>(
      "EvolvedSpaceFromCharacteristicFields");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.fd.Ccz4.Tags", "[Unit][Evolution]") {
  test_simple_tags<double, 3, ArbitraryFrame>();
  test_simple_tags<DataVector, 3, ArbitraryFrame>();
}
