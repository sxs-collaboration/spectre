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
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace {
template <size_t DerivOrder>
void test(const gsl::not_null<FunctionsOfTime::FunctionOfTime*> f_of_t,
          const gsl::not_null<FunctionsOfTime::PiecewisePolynomial<DerivOrder>*>
              f_of_t_derived,
          double t, const double dt, const double final_time) {
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
    double t, const double dt, const double final_time) {
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
void test_within_roundoff(const FunctionsOfTime::FunctionOfTime& f_of_t) {
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

class quartic {
 public:
  quartic(const std::array<DataVector, 5> coeffs) : coeffs_(coeffs) {}
  template <size_t MaxDerivReturned>
  std::array<DataVector, MaxDerivReturned + 1> func_and_derivs(double t) const {
    if constexpr (0 == MaxDerivReturned) {
      return {{coeffs_[0] +
               t * (coeffs_[1] +
                    t * (coeffs_[2] + t * (coeffs_[3] + t * coeffs_[4])))}};
    } else if constexpr (1 == MaxDerivReturned) {
      return {
          {coeffs_[0] +
               t * (coeffs_[1] +
                    t * (coeffs_[2] + t * (coeffs_[3] + t * coeffs_[4]))),
           coeffs_[1] + t * (2.0 * coeffs_[2] +
                             t * (3.0 * coeffs_[3] + t * 4.0 * coeffs_[4]))}};
    } else {
      static_assert(2 == MaxDerivReturned, "Unimplemented");
      return {
          {coeffs_[0] +
               t * (coeffs_[1] +
                    t * (coeffs_[2] + t * (coeffs_[3] + t * coeffs_[4]))),
           coeffs_[1] + t * (2.0 * coeffs_[2] +
                             t * (3.0 * coeffs_[3] + t * 4.0 * coeffs_[4])),
           2.0 * coeffs_[2] + t * (6.0 * coeffs_[3] + t * 12.0 * coeffs_[4])}};
    }
  }

 private:
  std::array<DataVector, 5> coeffs_;
};

template <size_t MaxDeriv>
void check_func_and_derivs(
    const FunctionsOfTime::PiecewisePolynomial<MaxDeriv>& f_of_t,
    const quartic& q) {
  const double t = 0.5;
  CHECK(f_of_t.func(t) == q.func_and_derivs<0>(t));
  CHECK(f_of_t.func_and_deriv(t) == q.func_and_derivs<1>(t));
  CHECK(f_of_t.func_and_2_derivs(t) == q.func_and_derivs<2>(t));
}

void test_func_and_derivs() {
  const double t0 = 0.0;
  const double tx = 1.0;
  DataVector c0{0.125, 0.25};
  DataVector c1{0.375, 0.5};
  DataVector c2{0.625, 0.75};
  DataVector c3{0.875, 1.125};
  DataVector c4{1.25, 1.375};
  DataVector zero{0.0, 0.0};
  check_func_and_derivs(
      FunctionsOfTime::PiecewisePolynomial<0>(t0, {{c0}}, tx),
      quartic(std::array<DataVector, 5>{{c0, zero, zero, zero, zero}}));
  check_func_and_derivs(
      FunctionsOfTime::PiecewisePolynomial<1>(t0, {{c0, c1}}, tx),
      quartic(std::array<DataVector, 5>{{c0, c1, zero, zero, zero}}));
  check_func_and_derivs(
      FunctionsOfTime::PiecewisePolynomial<2>(t0, {{c0, c1, 2.0 * c2}}, tx),
      quartic(std::array<DataVector, 5>{{c0, c1, c2, zero, zero}}));
  check_func_and_derivs(
      FunctionsOfTime::PiecewisePolynomial<3>(
          t0, {{c0, c1, 2.0 * c2, 6.0 * c3}}, tx),
      quartic(std::array<DataVector, 5>{{c0, c1, c2, c3, zero}}));
  check_func_and_derivs(
      FunctionsOfTime::PiecewisePolynomial<4>(
          t0, {{c0, c1, 2.0 * c2, 6.0 * c3, 24.0 * c4}}, tx),
      quartic(std::array<DataVector, 5>{{c0, c1, c2, c3, c4}}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.PiecewisePolynomial",
                  "[Domain][Unit]") {
  FunctionsOfTime::register_derived_with_charm();
  test_func_and_derivs();
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

    f_of_t.update(2.0, {12.0, 0.0}, 2.1);
    f_of_t2.update(2.0, {12.0, 0.0}, 2.1);

    test_within_roundoff<deriv_order>(f_of_t);
    test_within_roundoff<deriv_order>(f_of_t2);

    INFO("Test stream operator.");

    CHECK(get_output(f_of_t) ==
          "t=1: (1,1) (3,2) (3,1) (1,0)\n"
          "t=2: (8,4) (12,4) (6,1) (2,0)");
  }
  {
    INFO("Test evaluation at update time.");
    FunctionsOfTime::PiecewisePolynomial<0> f_of_t(1.0, {{{1.0, 2.0}}}, 2.0);

    CHECK(f_of_t.func(2.0)[0] == DataVector{1.0, 2.0});
    f_of_t.update(2.0, {3.0, 4.0}, 2.1);
    CHECK(f_of_t.func(2.0)[0] == DataVector{1.0, 2.0});
  }

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func,
                                                                 0.1);
        f_of_t.update(2.0, {6.0, 0.0}, 2.1);
        f_of_t.update(1.0, {6.0, 0.0}, 2.1);
      }()),
      Catch::Contains(
          "t must be increasing from call to call. Attempted to update "
          "at time 1") and
          Catch::Contains(", which precedes the previous update time of 2"));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func,
                                                                 0.1);
        f_of_t.update(1.0, {6.0, 0.0}, 2.1);
        f_of_t.update(2.0, {6.0, 0.0}, 1.1);
      }()),
      Catch::Contains(
          "expiration_time must be nondecreasing from call to call. "
          "Attempted to change expiration time to 1.1") and
          Catch::Contains(
              ", which precedes the previous expiration time of 2.1"));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func,
                                                                 2.0);
        f_of_t.update(1.0, {6.0, 0.0}, 2.1);
      }()),
      Catch::Contains("Attempt to update PiecewisePolynomial at a time 1") and
          Catch::Contains(
              " that is earlier than the previous expiration time of 2"));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func,
                                                                 2.0);
        f_of_t.update(2.5, {6.0, 0.0}, 2.2);
      }()),
      Catch::Contains(
          "Attempt to set the expiration time of PiecewisePolynomial to "
          "a value 2.2") and
          Catch::Contains(" that is earlier than the current time 2.5"));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func,
                                                                 2.0);
        f_of_t.reset_expiration_time(1.5);
        f_of_t.update(2.5, {6.0, 0.0}, 2.2);
      }()),
      Catch::Contains("Attempted to change expiration time to 1.5") and
          Catch::Contains(
              ", which precedes the previous expiration time of 2"));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func,
                                                                 0.1);
        f_of_t.update(1.0, {6.0, 0.0, 0.0}, 1.1);
      }()),
      Catch::Contains(
          "the number of components trying to be updated (3) does not match "
          "the number of components (2) in the PiecewisePolynomial."));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func,
                                                                 2.0);
        f_of_t.update(2.0, {6.0, 0.0}, 2.1);
        f_of_t.func(0.5);
      }()),
      Catch::Contains("requested time 5.0") and
          Catch::Contains(" precedes earliest time 1"));

  CHECK_THROWS_WITH(
      ([]() {
        // two component system (x**3 and x**2)
        constexpr size_t deriv_order = 3;
        const std::array<DataVector, deriv_order + 1> init_func{
            {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
        FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func,
                                                                 2.0);
        f_of_t.func(2.2);
      }()),
      Catch::Contains(
          "Attempt to evaluate PiecewisePolynomial at a time 2.2") and
          Catch::Contains(" that is after the expiration time 2"));
}
}  // namespace domain
