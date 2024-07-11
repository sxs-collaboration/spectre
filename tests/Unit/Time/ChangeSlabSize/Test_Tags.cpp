// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Time.ChangeSlabSize.Tags", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::ChangeSlabSize::NewSlabSize>(
      "NewSlabSize");
  TestHelpers::db::test_simple_tag<
      Tags::ChangeSlabSize::NumberOfExpectedMessages>(
      "NumberOfExpectedMessages");
  TestHelpers::db::test_simple_tag<Tags::ChangeSlabSize::SlabSizeGoal>(
      "SlabSizeGoal");
}
