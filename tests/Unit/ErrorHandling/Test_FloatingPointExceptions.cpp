// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <cmath>
#include <limits>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "tests/Unit/TestHelpers.hpp"

// [[OutputRegex, Floating point exception!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.Invalid",
                  "[ErrorHandling][Unit]") {
  ERROR_TEST();
  enable_floating_point_exceptions();
  // clang-tidy: Value is never read
  double invalid = sqrt(-1.0); //NOLINT
  CHECK(true);
}

// [[OutputRegex, Floating point exception!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.Overflow",
                  "[ErrorHandling][Unit]") {
  ERROR_TEST();
  enable_floating_point_exceptions();
  volatile double overflow = std::numeric_limits<double>::max();
  overflow *= 1.0e300;
  CHECK(true);
}

// [[OutputRegex, Floating point exception!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.DivByZero",
                  "[ErrorHandling][Unit]") {
  ERROR_TEST();
  enable_floating_point_exceptions();
  volatile double div_by_zero = 1.0;
  div_by_zero /= 0.0;
  CHECK(true);
}

SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.Disable",
                  "[ErrorHandling][Unit]") {
  enable_floating_point_exceptions();
  disable_floating_point_exceptions();
  // clang-tidy: Value is never read
  double invalid = sqrt(-1.0); //NOLINT
  volatile double overflow = std::numeric_limits<double>::max();
  overflow *= 1.0e300;
  volatile double div_by_zero = 1.0;
  div_by_zero /= 0.0;
  CHECK(true);
}
