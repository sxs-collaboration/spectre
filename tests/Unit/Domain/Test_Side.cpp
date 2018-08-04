// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Domain/Side.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Side", "[Domain][Unit]") {
  Side side_lower = Side::Lower;
  CHECK(opposite(side_lower) == Side::Upper);
  CHECK(opposite(opposite(side_lower)) == Side::Lower);
  CHECK(get_output(side_lower) == "Lower");
  CHECK(get_output(Side::Upper) == "Upper");
}
