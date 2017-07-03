// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <charm++.h>

#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("TestFramework.Approx", "[Unit]") {
  /// [approx_test]
  CHECK(1.0 == approx(1.0 + 1e-15));
  CHECK(1.0 != approx(1.0 + 1e-13));
  /// [approx_test]
  // also check numbers that are not order unity, but do not include in example
  CHECK(1e-10 == approx(1e-10 + 1e-25).scale(1e-10));
  CHECK(1e-10 == approx(1e-10 + 1e-23));
  CHECK(1e-10 != approx(1e-10 + 1e-23).scale(1e-10));
  CHECK(1e+10 == approx(1e+10 + 1e-5));
  CHECK(1e+10 != approx(1e+10 + 1e-3));

  /// [approx_default]
  CHECK(sin(M_PI / 4.0) == approx(cos(M_PI / 4.0)));
  /// [approx_default]
  /// [approx_single_custom]
  // This check needs tolerance 1e-12 for X reason.
  CHECK(1.0 == approx(1.0 + 5e-13).epsilon(1e-12));
  /// [approx_single_custom]
  /// [approx_new_custom]
  // The checks in this test need tolerance 1d-12 for X reason.
  Approx my_approx = Approx::custom().epsilon(1e-12);
  CHECK(1.0 == my_approx(1.0 + 5e-13));
  CHECK(1.0 != my_approx(1.0 + 5e-12));
  /// [approx_new_custom]
}

/// [error_test]
// [[OutputRegex, I failed]]
SPECTRE_TEST_CASE("TestFramework.Abort", "[Unit]") {
  ERROR_TEST();
  /// [error_test]
  CkAbort("I failed");
}
