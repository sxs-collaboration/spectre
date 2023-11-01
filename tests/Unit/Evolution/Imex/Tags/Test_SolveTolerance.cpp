// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Imex/Tags/SolveTolerance.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Tags.SolveTolerance",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<imex::Tags::SolveTolerance>(
      "SolveTolerance");
  CHECK(TestHelpers::test_option_tag<imex::OptionTags::SolveTolerance>(
            "0.125") == 0.125);
}
