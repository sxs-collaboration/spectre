// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<gh::Tags::Pi<Dim, Frame>>("Pi");
  TestHelpers::db::test_simple_tag<gh::Tags::Phi<Dim, Frame>>("Phi");
  TestHelpers::db::test_simple_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma0>("ConstraintGamma0");
  TestHelpers::db::test_simple_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma1>("ConstraintGamma1");
  TestHelpers::db::test_simple_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma2>("ConstraintGamma2");
  TestHelpers::db::test_simple_tag<gh::Tags::GaugeH<Dim, Frame>>("GaugeH");
  TestHelpers::db::test_simple_tag<gh::Tags::SpacetimeDerivGaugeH<Dim, Frame>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_simple_tag<gh::Tags::InitialGaugeH<Dim, Frame>>(
      "InitialGaugeH");
  TestHelpers::db::test_simple_tag<
      gh::Tags::SpacetimeDerivInitialGaugeH<Dim, Frame>>(
      "SpacetimeDerivInitialGaugeH");
  TestHelpers::db::test_simple_tag<gh::Tags::VSpacetimeMetric<Dim, Frame>>(
      "VSpacetimeMetric");
  TestHelpers::db::test_simple_tag<gh::Tags::VZero<Dim, Frame>>("VZero");
  TestHelpers::db::test_simple_tag<gh::Tags::VPlus<Dim, Frame>>("VPlus");
  TestHelpers::db::test_simple_tag<gh::Tags::VMinus<Dim, Frame>>("VMinus");
  TestHelpers::db::test_simple_tag<gh::Tags::CharacteristicSpeeds<Dim, Frame>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<gh::Tags::CharacteristicFields<Dim, Frame>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      gh::Tags::EvolvedFieldsFromCharacteristicFields<Dim, Frame>>(
      "EvolvedFieldsFromCharacteristicFields");
  TestHelpers::db::test_simple_tag<gh::Tags::GaugeConstraint<Dim, Frame>>(
      "GaugeConstraint");
  TestHelpers::db::test_simple_tag<gh::Tags::FConstraint<Dim, Frame>>(
      "FConstraint");
  TestHelpers::db::test_simple_tag<gh::Tags::TwoIndexConstraint<Dim, Frame>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_simple_tag<gh::Tags::ThreeIndexConstraint<Dim, Frame>>(
      "ThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<gh::Tags::FourIndexConstraint<Dim, Frame>>(
      "FourIndexConstraint");

  TestHelpers::db::test_simple_tag<gh::Tags::ConstraintEnergy<Dim, Frame>>(
      "ConstraintEnergy");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Tags",
                  "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame>();
  test_simple_tags<2, ArbitraryFrame>();
  test_simple_tags<3, ArbitraryFrame>();
}
