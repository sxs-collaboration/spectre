// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Psi>("Psi");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Pi>("Pi");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Phi<3>>("Phi");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::ConstraintGamma1>(
      "ConstraintGamma1");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::ConstraintGamma2>(
      "ConstraintGamma2");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Tags::OneIndexConstraint<3>>("OneIndexConstraint");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Tags::TwoIndexConstraint<3>>("TwoIndexConstraint");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::VPsi>("VPsi");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::VZero<3>>("VZero");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::VPlus>("VPlus");
  TestHelpers::db::test_simple_tag<CurvedScalarWave::Tags::VMinus>("VMinus");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Tags::CharacteristicSpeeds<3>>("CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Tags::CharacteristicFields<3>>("CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<3>>(
      "EvolvedFieldsFromCharacteristicFields");
}
