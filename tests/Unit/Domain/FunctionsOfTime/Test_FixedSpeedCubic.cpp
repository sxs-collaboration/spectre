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

namespace {
void test(
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>& f_of_t,
    const double initial_function_value, const double initial_time,
    const double velocity, const double decay_timescale) noexcept {
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

  // test time_bounds function
  const auto t_bounds = f_of_t->time_bounds();
  CHECK(t_bounds[0] == initial_time);
  CHECK(t_bounds[1] == std::numeric_limits<double>::max());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.FixedSpeedCubic",
                  "[Domain][Unit]") {
  using FixedSpeedCubic = domain::FunctionsOfTime::FixedSpeedCubic;

  domain::FunctionsOfTime::register_derived_with_charm();

  constexpr double initial_function_value{1.0};
  constexpr double initial_time{10.0};
  constexpr double velocity{0.4};
  constexpr double decay_timescale{5.0};

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

  {
    INFO("Test operator==");
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
}

// [[OutputRegex, FixedSpeedCubic denominator should not be zero]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.FunctionsOfTime.FixedSpeedCubic.CheckDenom",
    "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  constexpr double initial_function_value = 1.0;
  constexpr double initial_time = 0.0;
  constexpr double velocity = -0.1;
  constexpr double decay_timescale = 0.0;
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
          initial_function_value, initial_time, velocity, decay_timescale);
  test(f_of_t, initial_function_value, initial_time, velocity, decay_timescale);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
