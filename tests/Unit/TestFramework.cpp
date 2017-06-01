// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <charm++.h>

// [[OutputRegex, I failed]]
TEST_CASE("TestFramework.Abort", "[Unit]") { CkAbort("I failed"); }
