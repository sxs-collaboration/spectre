// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test(
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>& f_of_t,
    const double match_time, const double f_t0, const double dtf_t0,
    const double d2tf_t0, const double A) noexcept {
  // check that values agree at the matching time
  const auto lambdas0 = f_of_t->func_and_2_derivs(match_time);
  CHECK(approx(lambdas0[0][0]) == f_t0);
  CHECK(approx(lambdas0[1][0]) == dtf_t0);
  CHECK(approx(lambdas0[2][0]) == d2tf_t0);

  const auto lambdas1 = f_of_t->func_and_deriv(match_time);
  CHECK(approx(lambdas1[0][0]) == f_t0);
  CHECK(approx(lambdas1[1][0]) == dtf_t0);

  const auto lambdas2 = f_of_t->func(match_time);
  CHECK(approx(lambdas2[0][0]) == f_t0);

  // check that asymptotic values approach f = A = const.
  const auto lambdas3 = f_of_t->func_and_2_derivs(1.0e5);
  CHECK(approx(lambdas3[0][0]) == A);
  CHECK(approx(lambdas3[1][0]) == 0.0);
  CHECK(approx(lambdas3[2][0]) == 0.0);

  const auto lambdas4 = f_of_t->func_and_deriv(1.0e5);
  CHECK(approx(lambdas4[0][0]) == A);
  CHECK(approx(lambdas4[1][0]) == 0.0);

  const auto lambdas5 = f_of_t->func(1.0e5);
  CHECK(approx(lambdas5[0][0]) == A);

  // test time_bounds function
  const auto t_bounds = f_of_t->time_bounds();
  CHECK(t_bounds[0] == 10.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.SettleToConstant",
                  "[Domain][Unit]") {
  using SettleToConstant = domain::FunctionsOfTime::SettleToConstant;

  PUPable_reg(SettleToConstant);

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

  INFO("Test with base class construction.");
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<domain::FunctionsOfTime::SettleToConstant>(
          init_func, match_time, decay_time);
  test(f_of_t, match_time, f_t0, dtf_t0, d2tf_t0, A);

  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t2 =
      serialize_and_deserialize(f_of_t);
  test(f_of_t2, match_time, f_t0, dtf_t0, d2tf_t0, A);

  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t3 =
      f_of_t->get_clone();
  test(f_of_t3, match_time, f_t0, dtf_t0, d2tf_t0, A);

  {
    INFO("Test operator==");
    CHECK(SettleToConstant{init_func, match_time, decay_time} ==
          SettleToConstant{init_func, match_time, decay_time});
    CHECK_FALSE(SettleToConstant{init_func, match_time, decay_time} !=
                SettleToConstant{init_func, match_time, decay_time});

    CHECK(SettleToConstant{init_func, match_time, decay_time} !=
          SettleToConstant{init_func, match_time, 2.0 * decay_time});
    CHECK_FALSE(SettleToConstant{init_func, match_time, decay_time} ==
                SettleToConstant{init_func, match_time, 2.0 * decay_time});

    CHECK(SettleToConstant{init_func, match_time, decay_time} !=
          SettleToConstant{init_func, 2.0 * match_time, decay_time});
    CHECK_FALSE(SettleToConstant{init_func, match_time, decay_time} ==
                SettleToConstant{init_func, 2.0 * match_time, decay_time});

    CHECK(SettleToConstant{init_func, match_time, decay_time} !=
          SettleToConstant{{{init_func[0] + 1.0, init_func[1], init_func[2]}},
                           match_time,
                           decay_time});
    CHECK_FALSE(
        SettleToConstant{init_func, match_time, decay_time} ==
        SettleToConstant{{{init_func[0] + 1.0, init_func[1], init_func[2]}},
                         match_time,
                         decay_time});

    CHECK(SettleToConstant{init_func, match_time, decay_time} !=
          SettleToConstant{{{init_func[0], init_func[1] + 1.0, init_func[2]}},
                           match_time,
                           decay_time});
    CHECK_FALSE(
        SettleToConstant{init_func, match_time, decay_time} ==
        SettleToConstant{{{init_func[0], init_func[1] + 1.0, init_func[2]}},
                         match_time,
                         decay_time});

    CHECK(SettleToConstant{init_func, match_time, decay_time} !=
          SettleToConstant{{{init_func[0], init_func[1], init_func[2] + 1.0}},
                           match_time,
                           decay_time});
    CHECK_FALSE(
        SettleToConstant{init_func, match_time, decay_time} ==
        SettleToConstant{{{init_func[0], init_func[1], init_func[2] + 1.0}},
                         match_time,
                         decay_time});
  }
}
