// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>

#include "Utilities/MemoryHelpers.hpp"

// [[OutputRegex, Failed to allocate memory]]
SPECTRE_TEST_CASE("Unit.Utilities.allocation_failure", "[Unit][Time]") {
  ERROR_TEST();
  std::vector<int>(1000000000000000);
}
