// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame>
void test_simple_tags() {
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::Pi<Dim, Frame>>() == "Pi");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::Phi<Dim, Frame>>() == "Phi");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::ConstraintGamma0>() ==
        "ConstraintGamma0");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::ConstraintGamma1>() ==
        "ConstraintGamma1");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::ConstraintGamma2>() ==
        "ConstraintGamma2");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>>() ==
        "GaugeH");
  CHECK(db::tag_name<
            GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>() ==
        "SpacetimeDerivGaugeH");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame>>() ==
        "InitialGaugeH");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::UPsi<Dim, Frame>>() == "UPsi");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::UZero<Dim, Frame>>() ==
        "UZero");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::UPlus<Dim, Frame>>() ==
        "UPlus");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::UMinus<Dim, Frame>>() ==
        "UMinus");
  CHECK(db::tag_name<
            GeneralizedHarmonic::Tags::CharacteristicSpeeds<Dim, Frame>>() ==
        "CharacteristicSpeeds");
  CHECK(db::tag_name<
            GeneralizedHarmonic::Tags::CharacteristicFields<Dim, Frame>>() ==
        "CharacteristicFields");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::
                         EvolvedFieldsFromCharacteristicFields<Dim, Frame>>() ==
        "EvolvedFieldsFromCharacteristicFields");
  CHECK(
      db::tag_name<GeneralizedHarmonic::Tags::GaugeConstraint<Dim, Frame>>() ==
      "GaugeConstraint");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::FConstraint<Dim, Frame>>() ==
        "FConstraint");
  CHECK(db::tag_name<
            GeneralizedHarmonic::Tags::TwoIndexConstraint<Dim, Frame>>() ==
        "TwoIndexConstraint");
  CHECK(db::tag_name<
            GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, Frame>>() ==
        "ThreeIndexConstraint");
  CHECK(db::tag_name<
            GeneralizedHarmonic::Tags::FourIndexConstraint<Dim, Frame>>() ==
        "FourIndexConstraint");
  CHECK(
      db::tag_name<GeneralizedHarmonic::Tags::ConstraintEnergy<Dim, Frame>>() ==
      "ConstraintEnergy");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::GaugeHRollOnStartTime>() ==
        "GaugeHRollOnStartTime");
  CHECK(db::tag_name<GeneralizedHarmonic::Tags::GaugeHRollOnTimeWindow>() ==
        "GaugeHRollOnTimeWindow");
  CHECK(
      db::tag_name<
          GeneralizedHarmonic::Tags::GaugeHSpatialWeightDecayWidth<Frame>>() ==
      "GaugeHSpatialWeightDecayWidth");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Tags",
                  "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame>();
  test_simple_tags<2, ArbitraryFrame>();
  test_simple_tags<3, ArbitraryFrame>();
}
