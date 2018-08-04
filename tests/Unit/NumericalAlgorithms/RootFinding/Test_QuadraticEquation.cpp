// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

// [[OutputRegex, Assumes that there are two real roots]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.RealRoots.TwoRealRoots",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  real_roots(1.0, -3.0, 3.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Assumes that there are two real roots]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.PositiveRoot.TwoRealRoots",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  positive_root(1.0, -3.0, 3.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Ensures violated: x0 <= 0.0 and x1 >= 0.0]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.PositiveRoot.TwoPosRoots",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  positive_root(1.0, -3.0, 2.0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.OnePositiveRoot",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  CHECK(approx(6.31662479035539985) == positive_root(1.0, -6.0, -2.0));
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.RealRoots",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  auto roots = real_roots(2.0, -11.0, 5.0);
  CHECK(approx(0.5) == roots[0]);
  CHECK(approx(5.0) == roots[1]);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.Accuracy",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  auto small_positive = real_roots(1e-8, 1.0 - 1e-16, -1e-8);
  std::sort(small_positive.begin(), small_positive.end());
  CHECK(approx(-1e8) == small_positive[0]);
  CHECK(approx(1e-8) == small_positive[1]);

  auto small_negative = real_roots(1e-8, -(1.0 - 1e-16), -1e-8);
  std::sort(small_negative.begin(), small_negative.end());
  CHECK(approx(-1e-8) == small_negative[0]);
  CHECK(approx(1e8) == small_negative[1]);
}
