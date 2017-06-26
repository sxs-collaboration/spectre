// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Numerical/RootFinding/QuadraticEquation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// [[OutputRegex, Assumes that there are two real roots]]
[[noreturn]] TEST_CASE("Unit.Numerical.RootFinding.RealRoots.TwoRealRoots",
          "[Numerical][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  real_roots(1.0, -3.0, 3.0);
  ERROR("Bad test end");
#endif
}

// [[OutputRegex, Assumes that there are two real roots]]
[[noreturn]] TEST_CASE("Unit.Numerical.RootFinding.PositiveRoot.TwoRealRoots",
          "[Numerical][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  positive_root(1.0, -3.0, 3.0);
  ERROR("Bad test end");
#endif
}

// [[OutputRegex, Ensures violated: x0 <= 0.0 and x1 >= 0.0]]
[[noreturn]] TEST_CASE("Unit.Numerical.RootFinding.PositiveRoot.TwoPosRoots",
          "[Numerical][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  positive_root(1.0, -3.0, 2.0);
  ERROR("Bad test end");
#endif
}

TEST_CASE("Unit.Numerical.RootFinding.OnePositiveRoot",
          "[Numerical][RootFinding][Unit]") {
  CHECK(Approx(6.31662479035539985) == positive_root(1.0, -6.0, -2.0));
}

TEST_CASE("Unit.Functors.RealRoots", "[Numerical][RootFinding][Unit]") {
  auto roots = real_roots(2.0, -11.0, 5.0);
  CHECK(Approx(0.5) == roots[0]);
  CHECK(Approx(5.0) == roots[1]);
}
