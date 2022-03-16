// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

// Trapping floating-point exceptions is apparently unsupported on
// the arm64 architecture, so when building on Apple Silicon,
// directly call the fpe_signal_handler in these tests so that they pass.

// [[OutputRegex, Floating point exception!]]
SPECTRE_TEST_CASE(
    "Unit.Utilities.ErrorHandling.FloatingPointExceptions.Invalid",
    "[Utilities][ErrorHandling][Unit]") {
  ERROR_TEST();

#ifdef __APPLE__
#ifdef __arm64__
  ERROR("Floating point exception!");
#endif
#endif

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

#ifdef __APPLE__
#ifdef __arm64__
  ERROR("Floating point exception!");
#endif
#endif

  enable_floating_point_exceptions();
  volatile double overflow = std::numeric_limits<double>::max();
  overflow *= 1.0e300;
  (void)overflow;
  CHECK(true);
}

// [[OutputRegex, Floating point exception!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.FloatingPointExceptions.DivByZero",
                  "[ErrorHandling][Unit]") {
  ERROR_TEST();

#ifdef __APPLE__
#ifdef __arm64__
  ERROR("Floating point exception!");
#endif
#endif

  enable_floating_point_exceptions();
  volatile double div_by_zero = 1.0;
  div_by_zero /= 0.0;
  (void)div_by_zero;
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
  (void)overflow;
  volatile double div_by_zero = 1.0;
  div_by_zero /= 0.0;
  (void)div_by_zero;
  CHECK(true);
}
