// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"

namespace {
void test_simpletag() {
  TestHelpers::db::test_simple_tag<evolution::initial_data::Tags::InitialData>(
      "InitialData");
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.InitialDataUtilities.Tags.InitialData",
    "[Unit][PointwiseFunctions]") {
  test_simpletag();
}
