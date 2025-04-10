// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Domain/Structure/Side.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Structure.Side", "[Domain][Unit]") {
  Side side_lower = Side::Lower;
  CHECK(opposite(side_lower) == Side::Upper);
  CHECK(opposite(opposite(side_lower)) == Side::Lower);
  CHECK(opposite(Side::Self) == Side::Self);
  CHECK(get_output(side_lower) == "Lower");
  CHECK(get_output(Side::Upper) == "Upper");
  CHECK(get_output(Side::Uninitialized) == "Uninitialized");
  CHECK(get_output(Side::Self) == "Self");
  CHECK_THROWS_WITH(opposite(Side::Uninitialized),
                    Catch::Matchers::ContainsSubstring(
                        "Cannot get the opposite of Side::Uninitialized"));
}
