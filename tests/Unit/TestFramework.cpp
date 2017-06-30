// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <charm++.h>

#include "tests/Unit/TestHelpers.hpp"

/// [error_test]
// [[OutputRegex, I failed]]
TEST_CASE("TestFramework.Abort", "[Unit]") {
  ERROR_TEST();
  /// [error_test]
  CkAbort("I failed");
}
