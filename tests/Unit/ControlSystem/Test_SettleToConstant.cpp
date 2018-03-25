// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>

#include "ControlSystem/FunctionOfTime.hpp"
#include "ControlSystem/SettleToConstant.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.FunctionsOfTime.SettleToConstant",
                  "[ControlSystem][Unit]") {
  const double match_time = 10.0;
  const double decay_time = 5.0;
  // set init_func to f(t) = 5 + 2t + 3t^2
  const double f_t0 = 5.0 + 2.0 * match_time + 3.0 * square(match_time);
  const double dtf_t0 = 2.0 + 6.0 * match_time;
  const double d2tf_t0 = 6.0;
  // compute coefficients for check
  const double C = -(decay_time * d2tf_t0 + dtf_t0);
  const double B = decay_time * (C - dtf_t0);
  const double A = f_t0 - B;
  const std::array<DataVector, 3> init_func{{{f_t0}, {dtf_t0}, {d2tf_t0}}};
  const FunctionsOfTime::SettleToConstant f_of_t_derived(init_func, match_time,
                                                         decay_time);
  const FunctionOfTime& f_of_t = f_of_t_derived;

  // check that values agree at the matching time
  const auto& lambdas0 = f_of_t.func_and_2_derivs(match_time);
  CHECK(approx(lambdas0[0][0]) == f_t0);
  CHECK(approx(lambdas0[1][0]) == dtf_t0);
  CHECK(approx(lambdas0[2][0]) == d2tf_t0);

  const auto& lambdas1 = f_of_t.func_and_deriv(match_time);
  CHECK(approx(lambdas1[0][0]) == f_t0);
  CHECK(approx(lambdas1[1][0]) == dtf_t0);

  const auto& lambdas2 = f_of_t.func(match_time);
  CHECK(approx(lambdas2[0][0]) == f_t0);

  // check that asymptotic values approach f = A = const.
  const auto& lambdas3 = f_of_t.func_and_2_derivs(1.0e5);
  CHECK(approx(lambdas3[0][0]) == A);
  CHECK(approx(lambdas3[1][0]) == 0.0);
  CHECK(approx(lambdas3[2][0]) == 0.0);

  const auto& lambdas4 = f_of_t.func_and_deriv(1.0e5);
  CHECK(approx(lambdas4[0][0]) == A);
  CHECK(approx(lambdas4[1][0]) == 0.0);

  const auto& lambdas5 = f_of_t.func(1.0e5);
  CHECK(approx(lambdas5[0][0]) == A);

  // test time_bounds function
  const auto& t_bounds = f_of_t.time_bounds();
  CHECK(t_bounds[0] == 10.0);
}
