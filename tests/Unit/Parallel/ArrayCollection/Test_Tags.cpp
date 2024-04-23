// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/ArrayCollection/Tags/ElementCollection.hpp"

namespace Parallel {
SPECTRE_TEST_CASE("Unit.Parallel.ArrayCollection.Tags", "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<
      Tags::ElementCollection<3, void, void, void>>("ElementCollection");
}

}  // namespace Parallel
