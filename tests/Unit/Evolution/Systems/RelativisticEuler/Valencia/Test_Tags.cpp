// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Tags", "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<
      RelativisticEuler::Valencia::Tags::CharacteristicSpeeds<3>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<RelativisticEuler::Valencia::Tags::TildeD>(
      "TildeD");
  TestHelpers::db::test_simple_tag<RelativisticEuler::Valencia::Tags::TildeTau>(
      "TildeTau");
  TestHelpers::db::test_simple_tag<
      RelativisticEuler::Valencia::Tags::TildeS<3, Frame::Logical>>(
      "Logical_TildeS");
}
