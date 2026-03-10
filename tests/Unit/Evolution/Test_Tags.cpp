// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Tags/Filter.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Tags", "[Unit][Evolution]") {
  struct FilterListTag;
  TestHelpers::db::test_simple_tag<Filters::Tags::FilterList<FilterListTag>>(
      "FilterList");
}
