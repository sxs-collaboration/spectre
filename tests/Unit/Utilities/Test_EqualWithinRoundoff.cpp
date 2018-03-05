// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/EqualWithinRoundoff.hpp"

static_assert(equal_within_roundoff(1.0, 1.0 - 4.0e-16, 1.0e-15),
              "Failed testing EqualWithinRoundoff");
static_assert(not equal_within_roundoff(1.0, 1.0 - 4.0e-15, 1.0e-15),
              "Failed testing EqualWithinRoundoff");

static_assert(equal_within_roundoff(1.0e16, 1.0e16 - 1.0e1, 1.0e-15, 1.0e16),
              "Failed testing EqualWithinRoundoff");
static_assert(not equal_within_roundoff(1.0e16, 1.0e16 - 1.0e1, 1.0e-15, 1.0),
              "Failed testing EqualWithinRoundoff");

static_assert(not equal_within_roundoff(1.0, 1.0 - 1.0e-8, 1.0e-8, 0.0),
              "Failed testing EqualWithinRoundoff");
static_assert(equal_within_roundoff(1.0, 1.0 - 1.0e-8, 1.0e-8, 1.0),
              "Failed testing EqualWithinRoundoff");
