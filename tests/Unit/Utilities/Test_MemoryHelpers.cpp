// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>

#include "Utilities/MemoryHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.allocation_failure", "[Unit][Time]") {
  CHECK_THROWS_WITH((std::vector<int>(1000000000000000)),
                    Catch::Matchers::Contains("Failed to allocate memory"));
}
