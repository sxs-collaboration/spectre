// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<ScalarWave::Psi>("Psi");
  TestHelpers::db::test_simple_tag<ScalarWave::Pi>("Pi");
  TestHelpers::db::test_simple_tag<ScalarWave::Phi<3>>("Phi");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::ConstraintGamma2>(
      "ConstraintGamma2");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::OneIndexConstraint<3>>(
      "OneIndexConstraint");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::TwoIndexConstraint<3>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::VPsi>("VPsi");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::VZero<3>>("VZero");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::VPlus>("VPlus");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::VMinus>("VMinus");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::CharacteristicSpeeds<3>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<ScalarWave::Tags::CharacteristicFields<3>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      ScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<3>>(
      "EvolvedFieldsFromCharacteristicFields");
}
