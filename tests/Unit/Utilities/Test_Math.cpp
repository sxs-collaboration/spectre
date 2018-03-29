// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Utilities/Math.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Math", "[Unit][Utilities]") {
  // Test number_of_digits
  CHECK(2 == number_of_digits(10));
  CHECK(1 == number_of_digits(0));
  CHECK(1 == number_of_digits(-1));
  CHECK(1 == number_of_digits(9));
  CHECK(2 == number_of_digits(-99));
}
