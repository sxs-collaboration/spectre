// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <limits>

#include "ErrorHandling/FloatingPointExceptions.hpp"

// [[OutputRegex, Floating point exception!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.Invalid",
                  "[ErrorHandling][Unit]") {
  ERROR_TEST();
  enable_floating_point_exceptions();
  volatile double x = -1.0;
  volatile double invalid = sqrt(x);
  static_cast<void>(invalid);
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
  double x = -1.0;
  double invalid = sqrt(x);
  static_cast<void>(invalid);
  volatile double overflow = std::numeric_limits<double>::max();
  overflow *= 1.0e300;
  volatile double div_by_zero = 1.0;
  div_by_zero /= 0.0;
  CHECK(true);
}
