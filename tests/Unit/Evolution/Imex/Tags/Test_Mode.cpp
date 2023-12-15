// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Imex/Mode.hpp"
#include "Evolution/Imex/Tags/Mode.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Tags.Mode", "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<imex::Tags::Mode>("Mode");
  CHECK(TestHelpers::test_option_tag<imex::OptionTags::Mode>("SemiImplicit") ==
        imex::Mode::SemiImplicit);
}
