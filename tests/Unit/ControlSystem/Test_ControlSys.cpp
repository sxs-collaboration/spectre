// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "ControlSystem/FunctionVsTime.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.FunctionVsTime",
                  "[ControlSystem][Unit]") {
  double t = 0;
  // timestep must be an integer or power of 1/2,
  // otherwise roundoff accumulates and CHECK will fail.
  //
  double dt = 0.8;
  double FinalTime = 3.2;
  const constexpr size_t DerivOrderFvt = 3;

  // test two component system (x**3 and x**2)
  std::vector<std::array<double, DerivOrderFvt + 1>> FuncInit{{0, 0, 0, 6},
                                                              {0, 0, 2, 0}};
  FunctionVsTimeNthDeriv<DerivOrderFvt> Fvt(t, FuncInit);

  while (t < FinalTime) {
    double value1 = t * t * t;
    double value2 = t * t;

    // retrieve lambdas from Fvt
    auto phi = Fvt(t);
    // [component][deriv]; zero deriv index is the function.
    double lambda1 = phi[0][0];
    double lambda2 = phi[1][0];

    // CHECK(lambda1 == value1);
    // CHECK(lambda2 == value2);
    CHECK(approx(lambda1) == value1);
    CHECK(approx(lambda2) == value2);

    // increase time
    t += dt;
    // do update
    std::vector<double> maxd{6.0, 0.0};
    Fvt.update(t, maxd);
  }
}
