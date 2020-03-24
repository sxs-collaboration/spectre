// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::Pi<Dim, Frame>>(
      "Pi");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::Phi<Dim, Frame>>(
      "Phi");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::ConstraintGamma0>(
      "ConstraintGamma0");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::ConstraintGamma1>(
      "ConstraintGamma1");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::ConstraintGamma2>(
      "ConstraintGamma2");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>>("GaugeH");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame>>("InitialGaugeH");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim, Frame>>(
      "SpacetimeDerivInitialGaugeH");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::UPsi<Dim, Frame>>(
      "UPsi");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::UZero<Dim, Frame>>("UZero");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::UPlus<Dim, Frame>>("UPlus");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::UMinus<Dim, Frame>>("UMinus");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::CharacteristicSpeeds<Dim, Frame>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::CharacteristicFields<Dim, Frame>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::EvolvedFieldsFromCharacteristicFields<Dim,
                                                                       Frame>>(
      "EvolvedFieldsFromCharacteristicFields");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::GaugeConstraint<Dim, Frame>>(
      "GaugeConstraint");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::FConstraint<Dim, Frame>>("FConstraint");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::TwoIndexConstraint<Dim, Frame>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, Frame>>(
      "ThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::FourIndexConstraint<Dim, Frame>>(
      "FourIndexConstraint");

  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::ConstraintEnergy<Dim, Frame>>(
      "ConstraintEnergy");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::GaugeHRollOnStartTime>(
      "GaugeHRollOnStartTime");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::GaugeHRollOnTimeWindow>(
      "GaugeHRollOnTimeWindow");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::GaugeHSpatialWeightDecayWidth<Frame>>(
      "GaugeHSpatialWeightDecayWidth");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Tags",
                  "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame>();
  test_simple_tags<2, ArbitraryFrame>();
  test_simple_tags<3, ArbitraryFrame>();
}
