// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/InputSource.hpp"

namespace Parallel {
SPECTRE_TEST_CASE("Unit.Parallel.Tags.InputSource", "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<Tags::InputSource>("InputSource");
}
}  // namespace Parallel
