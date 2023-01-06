// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"

namespace {
struct TestArrayIndex;
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Tags.ArrayIndex", "[Unit][Parallel]") {
  TestHelpers::db::test_base_tag<Parallel::Tags::ArrayIndex>("ArrayIndex");
  TestHelpers::db::test_simple_tag<
      Parallel::Tags::ArrayIndexImpl<TestArrayIndex>>("ArrayIndex");
}
