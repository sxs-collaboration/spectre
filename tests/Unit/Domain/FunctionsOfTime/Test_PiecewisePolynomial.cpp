// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace {
template <size_t DerivOrder>
void test(const gsl::not_null<FunctionsOfTime::FunctionOfTime*> f_of_t,
          const gsl::not_null<FunctionsOfTime::PiecewisePolynomial<DerivOrder>*>
              f_of_t_derived,
          double t, const double dt, const double final_time) noexcept {
  const FunctionsOfTime::PiecewisePolynomial<DerivOrder> f_of_t_derived_copy =
      *f_of_t_derived;
  CHECK(*f_of_t_derived == f_of_t_derived_copy);
  while (t < final_time) {
    const auto lambdas0 = f_of_t->func_and_2_derivs(t);
    CHECK(approx(lambdas0[0][0]) == cube(t));
    CHECK(approx(lambdas0[0][1]) == square(t));
    CHECK(approx(lambdas0[1][0]) == 3.0 * square(t));
    CHECK(approx(lambdas0[1][1]) == 2.0 * t);
    CHECK(approx(lambdas0[2][0]) == 6.0 * t);
    CHECK(approx(lambdas0[2][1]) == 2.0);

    const auto lambdas1 = f_of_t->func_and_deriv(t);
    CHECK(approx(lambdas1[0][0]) == cube(t));
    CHECK(approx(lambdas1[0][1]) == square(t));
    CHECK(approx(lambdas1[1][0]) == 3.0 * square(t));
    CHECK(approx(lambdas1[1][1]) == 2.0 * t);

    const auto lambdas2 = f_of_t->func(t);
    CHECK(approx(lambdas2[0][0]) == cube(t));
    CHECK(approx(lambdas2[0][1]) == square(t));

    t += dt;
    f_of_t_derived->update(t, {6.0, 0.0}, t + dt);
    CHECK(*f_of_t_derived != f_of_t_derived_copy);
  }
  // test time_bounds function
  const auto t_bounds = f_of_t->time_bounds();
  CHECK(t_bounds[0] == 0.0);
  CHECK(t_bounds[1] == 4.8);
}

template <size_t DerivOrder>
void test_non_const_deriv(
    const gsl::not_null<FunctionsOfTime::FunctionOfTime*> f_of_t,
    const gsl::not_null<FunctionsOfTime::PiecewisePolynomial<DerivOrder>*>
        f_of_t_derived,
    double t, const double dt, const double final_time) noexcept {
  const FunctionsOfTime::PiecewisePolynomial<DerivOrder> f_of_t_derived_copy =
      *f_of_t_derived;
  CHECK(*f_of_t_derived == f_of_t_derived_copy);
  while (t < final_time) {
    t += dt;
    f_of_t_derived->update(t, {3.0 + t}, t + dt);
    CHECK(*f_of_t_derived != f_of_t_derived_copy);
  }
  t *= 1.0 + std::numeric_limits<double>::epsilon();
  const auto lambdas0 = f_of_t->func_and_2_derivs(t);
  CHECK(approx(lambdas0[0][0]) == 33.948);
  CHECK(approx(lambdas0[1][0]) == 19.56);
  CHECK(approx(lambdas0[2][0]) == 7.2);
  const auto lambdas1 = f_of_t->func_and_deriv(t);
  CHECK(approx(lambdas1[0][0]) == 33.948);
  CHECK(approx(lambdas1[1][0]) == 19.56);
  const auto lambdas2 = f_of_t->func(t);
  CHECK(approx(lambdas2[0][0]) == 33.948);

  CHECK(lambdas0.size() == 3);
  CHECK(lambdas1.size() == 2);
  CHECK(lambdas2.size() == 1);
}

template <size_t DerivOrder>
void test_within_roundoff(
    const FunctionsOfTime::FunctionOfTime& f_of_t) noexcept {
  const auto lambdas0 = f_of_t.func_and_2_derivs(1.0 - 5.0e-16);
  CHECK(approx(lambdas0[0][0]) == 1.0);
  CHECK(approx(lambdas0[1][0]) == 3.0);
  CHECK(approx(lambdas0[2][0]) == 6.0);
  CHECK(approx(lambdas0[0][1]) == 1.0);
  CHECK(approx(lambdas0[1][1]) == 2.0);
  CHECK(approx(lambdas0[2][1]) == 2.0);
  const auto lambdas1 = f_of_t.func_and_deriv(1.0 - 5.0e-16);
  CHECK(approx(lambdas1[0][0]) == 1.0);
  CHECK(approx(lambdas1[1][0]) == 3.0);
  CHECK(approx(lambdas1[0][1]) == 1.0);
  CHECK(approx(lambdas1[1][1]) == 2.0);
  const auto lambdas2 = f_of_t.func(1.0 - 5.0e-16);
  CHECK(approx(lambdas2[0][0]) == 1.0);
  CHECK(approx(lambdas2[0][1]) == 1.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.PiecewisePolynomial",
                  "[Domain][Unit]") {
  FunctionsOfTime::register_derived_with_charm();

  {
    INFO("Core test");
    double t = 0.0;
    const double dt = 0.6;
    const double final_time = 4.0;
    constexpr size_t deriv_order = 3;

    // test two component system (x**3 and x**2)
    const std::array<DataVector, deriv_order + 1> init_func{
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
    {
      INFO("Test with simple construction.");
      FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(t, init_func,
                                                               t + dt);
      FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t2 =
          serialize_and_deserialize(f_of_t);

      test(make_not_null(&f_of_t), make_not_null(&f_of_t), t, dt, final_time);
      test(make_not_null(&f_of_t2), make_not_null(&f_of_t2), t, dt, final_time);
    }
    {
      INFO("Test with base class construction.");
      std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t =
          std::make_unique<FunctionsOfTime::PiecewisePolynomial<deriv_order>>(
              t, init_func, t + dt);
      std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t2 =
          serialize_and_deserialize(f_of_t);
      std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t3 =
          f_of_t->get_clone();

      test(make_not_null(f_of_t.get()),
           make_not_null(
               dynamic_cast<FunctionsOfTime::PiecewisePolynomial<deriv_order>*>(
                   f_of_t.get())),
           t, dt, final_time);
      test(make_not_null(f_of_t2.get()),
           make_not_null(
               dynamic_cast<FunctionsOfTime::PiecewisePolynomial<deriv_order>*>(
                   f_of_t2.get())),
           t, dt, final_time);
      test(make_not_null(f_of_t3.get()),
           make_not_null(
               dynamic_cast<FunctionsOfTime::PiecewisePolynomial<deriv_order>*>(
                   f_of_t3.get())),
           t, dt, final_time);
    }
  }

  {
    INFO("Non-constant derivative test.");
    double t = 0.0;
    const double dt = 0.6;
    const double final_time = 4.0;
    constexpr size_t deriv_order = 2;

    // initally x**2, but update with non-constant 2nd deriv
    const std::array<DataVector, deriv_order + 1> init_func{
        {{0.0}, {0.0}, {2.0}}};
    {
      INFO("Test with simple construction.");
      FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(t, init_func,
                                                               t + dt);
      FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t2 =
          serialize_and_deserialize(f_of_t);
      test_non_const_deriv(make_not_null(&f_of_t), make_not_null(&f_of_t), t,
                           dt, final_time);
      test_non_const_deriv(make_not_null(&f_of_t2), make_not_null(&f_of_t2), t,
                           dt, final_time);
    }
    {
      INFO("Test with base class construction.");
      std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t =
          std::make_unique<FunctionsOfTime::PiecewisePolynomial<deriv_order>>(
              t, init_func, t + dt);
      std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t2 =
          serialize_and_deserialize(f_of_t);
      std::unique_ptr<FunctionsOfTime::FunctionOfTime> f_of_t3 =
          f_of_t->get_clone();

      test_non_const_deriv(
          make_not_null(f_of_t.get()),
          make_not_null(
              dynamic_cast<FunctionsOfTime::PiecewisePolynomial<deriv_order>*>(
                  f_of_t.get())),
          t, dt, final_time);
      test_non_const_deriv(
          make_not_null(f_of_t2.get()),
          make_not_null(
              dynamic_cast<FunctionsOfTime::PiecewisePolynomial<deriv_order>*>(
                  f_of_t2.get())),
          t, dt, final_time);
      test_non_const_deriv(
          make_not_null(f_of_t3.get()),
          make_not_null(
              dynamic_cast<FunctionsOfTime::PiecewisePolynomial<deriv_order>*>(
                  f_of_t3.get())),
          t, dt, final_time);
    }
  }
  {
    INFO("Test within roundoff.");
    constexpr size_t deriv_order = 3;
    const std::array<DataVector, deriv_order + 1> init_func{
        {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};

    FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func,
                                                             1.1);
    FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t2 =
        serialize_and_deserialize(f_of_t);

    f_of_t.update(2.0, {6.0, 0.0}, 2.1);
    f_of_t2.update(2.0, {6.0, 0.0}, 2.1);

    test_within_roundoff<deriv_order>(f_of_t);
    test_within_roundoff<deriv_order>(f_of_t2);
  }
  {
    INFO("Test evaluation at update time.");
    FunctionsOfTime::PiecewisePolynomial<0> f_of_t(1.0, {{{1.0, 2.0}}}, 2.0);

    CHECK(f_of_t.func(2.0)[0] == DataVector{1.0, 2.0});
    f_of_t.update(2.0, {3.0, 4.0}, 2.1);
    CHECK(f_of_t.func(2.0)[0] == DataVector{1.0, 2.0});
  }
}

// [[OutputRegex, t must be increasing from call to call. Attempted to update at
// time 1, which precedes the previous update time of 2.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.BadUpdateTime",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func, 0.1);
  f_of_t.update(2.0, {6.0, 0.0}, 2.1);
  f_of_t.update(1.0, {6.0, 0.0}, 2.1);
}

// [[OutputRegex, expiration_time must be nondecreasing from call to call.
// Attempted to change expiration time to 1\.1, which precedes the
// previous expiration time of 2\.1.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.BadExpiryTime",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func, 0.1);
  f_of_t.update(1.0, {6.0, 0.0}, 2.1);
  f_of_t.update(2.0, {6.0, 0.0}, 1.1);
}

// [[OutputRegex, Attempt to update PiecewisePolynomial at a time 1
// that is earlier than the previous expiration time of 2.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.BadUpdateGtExpiry",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func, 2.0);
  f_of_t.update(1.0, {6.0, 0.0}, 2.1);
}

// [[OutputRegex, Attempt to set the expiration time of PiecewisePolynomial
// to a value 2\.2 that is earlier than the current time 2\.5.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.BadUpdateLtExpiry",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func, 2.0);
  f_of_t.update(2.5, {6.0, 0.0}, 2.2);
}

// [[OutputRegex, Attempted to change expiration time to
// 1\.5, which precedes the previous expiration time of 2.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.BadResetExpiry",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func, 2.0);
  f_of_t.reset_expiration_time(1.5);
}

// [[OutputRegex, the number of components trying to be updated \(3\) does not
// match the number of components \(2\) in the PiecewisePolynomial.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.BadUpdateSize",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func, 0.1);
  f_of_t.update(1.0, {6.0, 0.0, 0.0}, 1.1);
}

// [[OutputRegex, requested time 0.5 precedes earliest time 1 of times.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.TimeOutOfRange",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func, 2.0);
  f_of_t.update(2.0, {6.0, 0.0}, 2.1);
  f_of_t.func(0.5);
}

// [[OutputRegex, Attempt to evaluate PiecewisePolynomial at a time 2\.2
// that is after the expiration time 2\.]]
SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.PiecewisePolynomial.TimeAfterExpiry",
    "[Domain][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func, 2.0);
  f_of_t.func(2.2);
}
}  // namespace domain
