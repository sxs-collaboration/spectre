// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <vector>

#include "Utilities/EqualWithinRoundoff.hpp"

static_assert(equal_within_roundoff(1.0, 1.0, 0.0),
              "Failed testing EqualWithinRoundoff");

static_assert(equal_within_roundoff(1.0, 1.0 - 4.0e-16, 1.0e-15),
              "Failed testing EqualWithinRoundoff");
static_assert(not equal_within_roundoff(1.0, 1.0 - 4.0e-15, 1.0e-15),
              "Failed testing EqualWithinRoundoff");

static_assert(equal_within_roundoff(1.0e16, 1.0e16 - 1.0e1, 1.0e-15, 1.0e16),
              "Failed testing EqualWithinRoundoff");
static_assert(not equal_within_roundoff(1.0e16, 1.0e16 - 2.0e1, 1.0e-15, 1.0),
              "Failed testing EqualWithinRoundoff");

static_assert(not equal_within_roundoff(1.0, 1.0 - 1.0e-8, 1.0e-8, 0.0),
              "Failed testing EqualWithinRoundoff");
static_assert(equal_within_roundoff(1.0, 1.0 - 1.0e-8, 1.0e-8, 1.0),
              "Failed testing EqualWithinRoundoff");

SPECTRE_TEST_CASE("Unit.Utilities.EqualWithinRoundoff", "[Unit][Utilities]") {
  CHECK(equal_within_roundoff(
      std::vector<double>{1.0, 1.0 - 4.0e-16, 1.0 + 5.0e-15}, 1.0));
  CHECK(equal_within_roundoff(
      1.0, std::vector<double>{1.0, 1.0 - 4.0e-16, 1.0 + 5.0e-15}));
  CHECK(equal_within_roundoff(
      std::array<double, 3>{{1.0, 1.0 - 4.0e-16, 1.0 + 5.0e-15}}, 1.0));
  CHECK(equal_within_roundoff(
      1.0, std::array<double, 3>{{1.0, 1.0 - 4.0e-16, 1.0 + 5.0e-15}}));
  CHECK(equal_within_roundoff(
      std::vector<double>{1.0, 1.0 - 4.0e-16, 1.0 + 5.0e-15},
      std::array<double, 3>{{1.0 - 4.0e-16, 1.0 + 5.0e-15, 1.0}}));
  CHECK(equal_within_roundoff(
      std::vector<double>{1.0, 1.0 - 4.0e-10, 1.0 + 5.0e-10},
      std::array<double, 3>{{1.0 - 4.0e-10, 1.0 + 5.0e-10, 1.0}}, 1.e-9));
  CHECK_FALSE(equal_within_roundoff(
      std::vector<double>{1.0, 1.0 - 4.0e-10, 1.0 + 5.0e-10},
      std::array<double, 3>{{1.0 - 4.0e-10, 1.0 + 5.0e-10, 1.0}}, 1.e-11));
}
