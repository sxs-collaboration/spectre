// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>

#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// [[OutputRegex, There are only 0 real roots]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.real_roots.no_real_roots",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  real_roots(1.0, -3.0, 3.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, There are only 0 real roots]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.positive_root.no_real_roots",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  positive_root(1.0, -3.0, 3.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, There are two positive roots]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.positive_root.two_positive_roots",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  positive_root(1.0, -3.0, 2.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.QuadraticEquation",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  // Positive root
  CHECK(approx(6.31662479035539985) == positive_root(1.0, -6.0, -2.0));

  // Real roots
  const auto roots = real_roots(2.0, -11.0, 5.0);
  CHECK(approx(0.5) == roots[0]);
  CHECK(approx(5.0) == roots[1]);

  // Check accuracy with small roots
  const auto small_positive = real_roots(1e-8, 1.0 - 1e-16, -1e-8);
  CHECK(approx(-1e8) == small_positive[0]);
  CHECK(approx(1e-8) == small_positive[1]);

  const auto small_negative = real_roots(1e-8, -(1.0 - 1e-16), -1e-8);
  CHECK(approx(-1e-8) == small_negative[0]);
  CHECK(approx(1e8) == small_negative[1]);
}
