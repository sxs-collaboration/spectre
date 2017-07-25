// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "Utilities/MakeArithmeticValue.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.MakeArithmeticValue", "[Utilities][Unit]") {
  CHECK(make_arithmetic_value(1.3, 8.3) == 8.3);
  CHECK(make_arithmetic_value(DataVector(8, 4.5), -2.3) == DataVector(8, -2.3));
}
