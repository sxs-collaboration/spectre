// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Utilities/Stringize.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Stringize", "[Utilities][Unit]") {
  CHECK(stringize(false) == "false");
  CHECK(stringize(true) == "true");
}
