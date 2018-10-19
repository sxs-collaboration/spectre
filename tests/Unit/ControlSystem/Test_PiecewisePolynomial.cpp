// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "ControlSystem/FunctionOfTime.hpp"
#include "ControlSystem/PiecewisePolynomial.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial",
                  "[ControlSystem][Unit]") {
  double t = 0.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  constexpr size_t deriv_order = 3;

  // test two component system (x**3 and x**2)
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t_derived(t,
                                                                   init_func);
  FunctionOfTime& f_of_t = f_of_t_derived;

  while (t < final_time) {
    const auto& lambdas0 = f_of_t.func_and_2_derivs(t);
    CHECK(approx(lambdas0[0][0]) == cube(t));
    CHECK(approx(lambdas0[0][1]) == square(t));
    CHECK(approx(lambdas0[1][0]) == 3.0 * square(t));
    CHECK(approx(lambdas0[1][1]) == 2.0 * t);
    CHECK(approx(lambdas0[2][0]) == 6.0 * t);
    CHECK(approx(lambdas0[2][1]) == 2.0);

    const auto& lambdas1 = f_of_t.func_and_deriv(t);
    CHECK(approx(lambdas1[0][0]) == cube(t));
    CHECK(approx(lambdas1[0][1]) == square(t));
    CHECK(approx(lambdas1[1][0]) == 3.0 * square(t));
    CHECK(approx(lambdas1[1][1]) == 2.0 * t);

    const auto& lambdas2 = f_of_t.func(t);
    CHECK(approx(lambdas2[0][0]) == cube(t));
    CHECK(approx(lambdas2[0][1]) == square(t));

    t += dt;
    f_of_t_derived.update(t, {6.0, 0.0});
  }
  // test time_bounds function
  const auto& t_bounds = f_of_t.time_bounds();
  CHECK(t_bounds[0] == 0.0);
  CHECK(t_bounds[1] == 4.2);
}

SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.NonConstDeriv",
    "[ControlSystem][Unit]") {
  double t = 0.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  constexpr size_t deriv_order = 2;

  // initally x**2, but update with non-constant 2nd deriv
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0}, {0.0}, {2.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t_derived(t,
                                                                   init_func);
  FunctionOfTime& f_of_t = f_of_t_derived;

  while (t < final_time) {
    t += dt;
    f_of_t_derived.update(t, {3.0 + t});
  }
  const auto& lambdas0 = f_of_t.func_and_2_derivs(t);
  CHECK(approx(lambdas0[0][0]) == 33.948);
  CHECK(approx(lambdas0[1][0]) == 19.56);
  CHECK(approx(lambdas0[2][0]) == 7.2);
  const auto& lambdas1 = f_of_t.func_and_deriv(t);
  CHECK(approx(lambdas1[0][0]) == 33.948);
  CHECK(approx(lambdas1[1][0]) == 19.56);
  const auto& lambdas2 = f_of_t.func(t);
  CHECK(approx(lambdas2[0][0]) == 33.948);

  CHECK(lambdas0.size() == 3);
  CHECK(lambdas1.size() == 2);
  CHECK(lambdas2.size() == 1);
}

SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.WithinRoundoff",
    "[ControlSystem][Unit]") {
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func);
  f_of_t.update(2.0, {6.0, 0.0});

  const auto& lambdas0 = f_of_t.func_and_2_derivs(1.0 - 5.0e-16);
  CHECK(approx(lambdas0[0][0]) == 1.0);
  CHECK(approx(lambdas0[1][0]) == 3.0);
  CHECK(approx(lambdas0[2][0]) == 6.0);
  CHECK(approx(lambdas0[0][1]) == 1.0);
  CHECK(approx(lambdas0[1][1]) == 2.0);
  CHECK(approx(lambdas0[2][1]) == 2.0);
  const auto& lambdas1 = f_of_t.func_and_deriv(1.0 - 5.0e-16);
  CHECK(approx(lambdas1[0][0]) == 1.0);
  CHECK(approx(lambdas1[1][0]) == 3.0);
  CHECK(approx(lambdas1[0][1]) == 1.0);
  CHECK(approx(lambdas1[1][1]) == 2.0);
  const auto& lambdas2 = f_of_t.func(1.0 - 5.0e-16);
  CHECK(approx(lambdas2[0][0]) == 1.0);
  CHECK(approx(lambdas2[0][1]) == 1.0);
}

// [[OutputRegex, t must be increasing from call to call. Attempted to update at
// time 1, which precedes the previous update time of 2.]]
SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.BadUpdateTime",
    "[ControlSystem][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func);
  f_of_t.update(2.0, {6.0, 0.0});
  f_of_t.update(1.0, {6.0, 0.0});
}

// [[OutputRegex, the number of components trying to be updated \(3\) does not
// match the number of components \(2\) in the PiecewisePolynomial.]]
SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.BadUpdateSize",
    "[ControlSystem][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func);
  f_of_t.update(1.0, {6.0, 0.0, 0.0});
}

// [[OutputRegex, requested time 0.5 precedes earliest time 1 of times.]]
SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.TimeOutOfRange",
    "[ControlSystem][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func);
  f_of_t.update(2.0, {6.0, 0.0});
  f_of_t.func(0.5);
}
