// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/Info.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.Tags.Info", "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<Parallel::Tags::Info>("Info");
}
