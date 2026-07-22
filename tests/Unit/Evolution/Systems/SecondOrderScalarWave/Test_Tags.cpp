// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SecondOrderScalarWave.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::Psi>("Psi");
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::Pi>("Pi");
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::Phi<3>>("Phi");
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::NormalDotPhi>(
      "NormalDotPhi");
  TestHelpers::db::test_simple_tag<
      SecondOrderScalarWave::Tags::PsiTimesNormal<3>>("PsiTimesNormal");
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::VZero<3>>(
      "VZero");
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::VPlus>("VPlus");
  TestHelpers::db::test_simple_tag<SecondOrderScalarWave::Tags::VMinus>(
      "VMinus");
  TestHelpers::db::test_simple_tag<
      SecondOrderScalarWave::Tags::CharacteristicSpeeds<3>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<
      SecondOrderScalarWave::Tags::CharacteristicFields<3>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      SecondOrderScalarWave::Tags::FieldsFromInverseCharacteristicTransform<3>>(
      "FieldsFromInverseCharacteristicTransform");
}
