// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"

namespace {
struct DummyType {};
struct DummyTag : db::SimpleTag {
  using type = DummyType;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.AnalyticData.Tags", "[Unit][PointwiseFunctions]") {
  TestHelpers::db::test_base_tag<Tags::AnalyticSolutionOrData>(
      "AnalyticSolutionOrData");
  TestHelpers::db::test_base_tag<Tags::AnalyticDataBase>("AnalyticDataBase");
  TestHelpers::db::test_simple_tag<Tags::AnalyticData<DummyType>>(
      "AnalyticData");
}
