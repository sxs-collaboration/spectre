// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/Metavariables.hpp"

namespace {
struct TestMetavariables;
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Tags.Metavariables", "[Unit][Parallel]") {
  TestHelpers::db::test_base_tag<Parallel::Tags::Metavariables>(
      "Metavariables");
  TestHelpers::db::test_simple_tag<
      Parallel::Tags::MetavariablesImpl<TestMetavariables>>("Metavariables");
}
