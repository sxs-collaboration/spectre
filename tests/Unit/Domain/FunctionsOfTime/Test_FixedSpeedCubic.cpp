// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
void test(
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>& f_of_t,
    const double initial_function_value, const double initial_time,
    const double velocity, const double decay_timescale) {
  // Check that at t == initial time, f == initial_function_value,
  // df = 0, and d2f = 0
  const auto lambdas0 = f_of_t->func_and_2_derivs(initial_time);
  CHECK(approx(lambdas0[0][0]) == initial_function_value);
  CHECK(approx(lambdas0[1][0]) == 0.0);
  CHECK(approx(lambdas0[2][0]) == 0.0);

  const auto lambdas1 = f_of_t->func_and_deriv(initial_time);
  CHECK(approx(lambdas1[0][0]) == initial_function_value);
  CHECK(approx(lambdas1[1][0]) == 0.0);

  const auto lambdas2 = f_of_t->func(initial_time);
  CHECK(approx(lambdas2[0][0]) == initial_function_value);

  // Check that asymptotic values approach df == velocity, d2f == 0
  const auto lambdas3 = f_of_t->func_and_2_derivs(decay_timescale * 1.0e7);
  CHECK(approx(lambdas3[1][0]) == velocity);
  CHECK(approx(lambdas3[2][0]) == 0.0);

  const auto lambdas4 = f_of_t->func_and_deriv(decay_timescale * 1.0e7);
  CHECK(approx(lambdas4[1][0]) == velocity);

  // Check some time in between where we can easily calculate the answer by
  // hand. Choosing dt = 1.0 is helpful because 1^n = 1
  const double check_time = initial_time + 1.0;
  const double square_decay_timescale = square(decay_timescale);
  const double denom = square_decay_timescale + 1.0;
  const auto lambdas5 = f_of_t->func_and_2_derivs(check_time);
  CHECK(approx(lambdas5[0][0]) ==
        initial_function_value + velocity * 1.0 / denom);
  CHECK(approx(lambdas5[1][0]) ==
        velocity * (3.0 * square_decay_timescale + 1.0) / square(denom));
  CHECK(approx(lambdas5[2][0]) == 2.0 * velocity * square_decay_timescale *
                                      (3.0 * square_decay_timescale - 1.0) /
                                      cube(denom));

  // test time_bounds function
  const auto t_bounds = f_of_t->time_bounds();
  CHECK(t_bounds[0] == initial_time);
  CHECK(t_bounds[1] == std::numeric_limits<double>::max());

  INFO("Test stream operator.");
  CHECK(
      get_output(*dynamic_cast<const domain::FunctionsOfTime::FixedSpeedCubic*>(
          f_of_t.get())) == "FixedSpeedCubic(t=10, f=1, v=0.4, tau^2=25)");
}

void test_function(const double initial_function_value,
                   const double initial_time, const double velocity,
                   const double decay_timescale) {
  INFO("Test with base class construction.");
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
          initial_function_value, initial_time, velocity, decay_timescale);
  test(f_of_t, initial_function_value, initial_time, velocity, decay_timescale);

  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t2 =
      serialize_and_deserialize(f_of_t);
  test(f_of_t2, initial_function_value, initial_time, velocity,
       decay_timescale);

  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t3 =
      f_of_t->get_clone();
  test(f_of_t3, initial_function_value, initial_time, velocity,
       decay_timescale);
}

void test_operator(const double initial_function_value,
                   const double initial_time, const double velocity,
                   const double decay_timescale) {
  INFO("Test operator==");
  using FixedSpeedCubic = domain::FunctionsOfTime::FixedSpeedCubic;
  CHECK(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        decay_timescale} ==
        FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        decay_timescale});
  CHECK_FALSE(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              decay_timescale} !=
              FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              decay_timescale});

  CHECK(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        decay_timescale} !=
        FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        2.0 * decay_timescale});
  CHECK_FALSE(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              decay_timescale} ==
              FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              2.0 * decay_timescale});

  CHECK(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        decay_timescale} !=
        FixedSpeedCubic{initial_function_value, initial_time, 2.0 * velocity,
                        decay_timescale});
  CHECK_FALSE(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              decay_timescale} ==
              FixedSpeedCubic{initial_function_value, initial_time,
                              2.0 * velocity, decay_timescale});

  CHECK(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        decay_timescale} !=
        FixedSpeedCubic{initial_function_value, 2.0 * initial_time, velocity,
                        decay_timescale});
  CHECK_FALSE(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              decay_timescale} ==
              FixedSpeedCubic{initial_function_value, 2.0 * initial_time,
                              velocity, decay_timescale});

  CHECK(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                        decay_timescale} !=
        FixedSpeedCubic{2.0 * initial_function_value, initial_time, velocity,
                        decay_timescale});
  CHECK_FALSE(FixedSpeedCubic{initial_function_value, initial_time, velocity,
                              decay_timescale} ==
              FixedSpeedCubic{2.0 * initial_function_value, initial_time,
                              velocity, decay_timescale});
}

void test_errors() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        constexpr double initial_function_value = 1.0;
        constexpr double initial_time = 0.0;
        constexpr double velocity = -0.1;
        constexpr double decay_timescale = 0.0;
        const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
            std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
                initial_function_value, initial_time, velocity,
                decay_timescale);
        test(f_of_t, initial_function_value, initial_time, velocity,
             decay_timescale);
      }()),
      Catch::Contains("FixedSpeedCubic denominator should not be zero"));
#endif

  CHECK_THROWS_WITH(
      ([]() {
        const double initial_function_value = 1.0;
        const double initial_time = 0.0;
        const double velocity = -0.1;
        const double decay_timescale = 0.0;
        const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
            std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
                initial_function_value, initial_time, velocity,
                decay_timescale);

        const double update_time = 1.0;
        const DataVector updated_deriv{};
        const double next_expr_time = 2.0;
        f_of_t->update(update_time, updated_deriv, next_expr_time);
      }()),
      Catch::Contains("Cannot update this FunctionOfTime"));

  CHECK_THROWS_WITH(
      ([]() {
        const double initial_function_value = 1.0;
        const double initial_time = 0.0;
        const double velocity = -0.1;
        const double decay_timescale = 0.0;
        const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
            std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
                initial_function_value, initial_time, velocity,
                decay_timescale);

        const double next_expr_time = 2.0;
        f_of_t->reset_expiration_time(next_expr_time);
      }()),
      Catch::Contains("Cannot reset expiration time of this FunctionOfTime"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.FixedSpeedCubic",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();

  constexpr double initial_function_value{1.0};
  constexpr double initial_time{10.0};
  constexpr double velocity{0.4};
  constexpr double decay_timescale{5.0};

  test_function(initial_function_value, initial_time, velocity,
                decay_timescale);
  test_operator(initial_function_value, initial_time, velocity,
                decay_timescale);
  test_errors();
}
