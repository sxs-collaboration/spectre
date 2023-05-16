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
  TestHelpers::db::test_simple_tag<gh::Tags::Pi<DataVector, Dim, Frame>>("Pi");
  TestHelpers::db::test_simple_tag<gh::Tags::Phi<DataVector, Dim, Frame>>(
      "Phi");
  TestHelpers::db::test_simple_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma0>("ConstraintGamma0");
  TestHelpers::db::test_simple_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma1>("ConstraintGamma1");
  TestHelpers::db::test_simple_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma2>("ConstraintGamma2");
  TestHelpers::db::test_simple_tag<gh::Tags::GaugeH<DataVector, Dim, Frame>>(
      "GaugeH");
  TestHelpers::db::test_simple_tag<
      gh::Tags::SpacetimeDerivGaugeH<DataVector, Dim, Frame>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_simple_tag<
      gh::Tags::InitialGaugeH<DataVector, Dim, Frame>>("InitialGaugeH");
  TestHelpers::db::test_simple_tag<
      gh::Tags::SpacetimeDerivInitialGaugeH<DataVector, Dim, Frame>>(
      "SpacetimeDerivInitialGaugeH");
  TestHelpers::db::test_simple_tag<
      gh::Tags::VSpacetimeMetric<DataVector, Dim, Frame>>("VSpacetimeMetric");
  TestHelpers::db::test_simple_tag<gh::Tags::VZero<DataVector, Dim, Frame>>(
      "VZero");
  TestHelpers::db::test_simple_tag<gh::Tags::VPlus<DataVector, Dim, Frame>>(
      "VPlus");
  TestHelpers::db::test_simple_tag<gh::Tags::VMinus<DataVector, Dim, Frame>>(
      "VMinus");
  TestHelpers::db::test_simple_tag<
      gh::Tags::CharacteristicSpeeds<DataVector, Dim, Frame>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<
      gh::Tags::CharacteristicFields<DataVector, Dim, Frame>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      gh::Tags::EvolvedFieldsFromCharacteristicFields<DataVector, Dim, Frame>>(
      "EvolvedFieldsFromCharacteristicFields");
  TestHelpers::db::test_simple_tag<
      gh::Tags::GaugeConstraint<DataVector, Dim, Frame>>("GaugeConstraint");
  TestHelpers::db::test_simple_tag<
      gh::Tags::FConstraint<DataVector, Dim, Frame>>("FConstraint");
  TestHelpers::db::test_simple_tag<
      gh::Tags::TwoIndexConstraint<DataVector, Dim, Frame>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_simple_tag<
      gh::Tags::ThreeIndexConstraint<DataVector, Dim, Frame>>(
      "ThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<
      gh::Tags::FourIndexConstraint<DataVector, Dim, Frame>>(
      "FourIndexConstraint");

  TestHelpers::db::test_simple_tag<
      gh::Tags::ConstraintEnergy<DataVector, Dim, Frame>>("ConstraintEnergy");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Tags",
                  "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame>();
  test_simple_tags<2, ArbitraryFrame>();
  test_simple_tags<3, ArbitraryFrame>();
}
