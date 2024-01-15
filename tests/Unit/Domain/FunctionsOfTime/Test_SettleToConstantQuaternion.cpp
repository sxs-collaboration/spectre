// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
void test(
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>& f_of_t,
    const double match_time, const double decay_time, const DataVector& f_t0,
    const DataVector& dtf_t0, const DataVector& d2tf_t0, const DataVector& A) {
  // check that values agree at the matching time
  const auto lambdas0 = f_of_t->func_and_2_derivs(match_time);
  CHECK_ITERABLE_APPROX(lambdas0[0], f_t0);
  CHECK_ITERABLE_APPROX(lambdas0[1], dtf_t0);
  CHECK_ITERABLE_APPROX(lambdas0[2], d2tf_t0);

  const auto lambdas1 = f_of_t->func_and_deriv(match_time);
  CHECK_ITERABLE_APPROX(lambdas1[0], f_t0);
  CHECK_ITERABLE_APPROX(lambdas1[1], dtf_t0);

  const auto lambdas2 = f_of_t->func(match_time);
  CHECK_ITERABLE_APPROX(lambdas2[0], f_t0);

  // check that asymptotic values approach f = A/|A| = const.
  const auto lambdas3 = f_of_t->func_and_2_derivs(1.0e5);
  const DataVector zero{0.0, 0.0, 0.0, 0.0};
  const DataVector normalized_A = A / norm(A);
  CHECK_ITERABLE_APPROX(lambdas3[0], normalized_A);
  CHECK_ITERABLE_APPROX(lambdas3[1], zero);
  CHECK_ITERABLE_APPROX(lambdas3[2], zero);

  const auto lambdas4 = f_of_t->func_and_deriv(1.0e5);
  CHECK_ITERABLE_APPROX(lambdas4[0], normalized_A);
  CHECK_ITERABLE_APPROX(lambdas4[1], zero);

  const auto lambdas5 = f_of_t->func(1.0e5);
  CHECK_ITERABLE_APPROX(lambdas5[0], normalized_A);

  // Check that function of time remains a unit quaternion
  const auto lambdas6 = f_of_t->func(match_time + decay_time * 0.34);
  const double lambdas6_norm = norm(lambdas6[0]);
  CHECK(approx(lambdas6_norm) == 1.0);

  // test time_bounds function
  const auto t_bounds = f_of_t->time_bounds();
  CHECK(t_bounds[0] == match_time);
  CHECK(t_bounds[1] == std::numeric_limits<double>::infinity());
  CHECK(f_of_t->expiration_after(match_time) ==
        std::numeric_limits<double>::infinity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.SettleToConstantQuaternion",
                  "[Domain][Unit]") {
  using SettleToConstantQuaternion =
      domain::FunctionsOfTime::SettleToConstantQuaternion;

  domain::FunctionsOfTime::register_derived_with_charm();

  const double match_time = 10.0;
  const double decay_time = 5.0;
  // Choose for an initial quaternion
  // q = (cos \theta/2, \hat{n} \sin \theta/2), where
  // \theta = \Omega(t-t_0) + \alpha(t-t_0)^2/2 + \phi
  // \hat{n}_x = \cos \Phi
  // \hat{n}_y = 0
  // \hat{n}_z = \sin \Phi
  // \Phi = \omega(t-t_0) + \varphi
  // This is an accelerating, precessing rotation.
  // For the constants, choose
  // \Omega = 0.1, \alpha=0.05, \omega = 0.01, \phi = 0.2, \varphi = 0.4
  const DataVector f_t0{0.9950041652780258, 0.09195266597143172, 0.0,
                        0.03887696361761665};
  const DataVector dtf_t0{-0.004991670832341408, 0.04543420663922331, 0.0,
                          0.02029317029135289};
  const DataVector d2tf_t0{-0.004983345829365769, 0.022284938333541247, 0.0,
                           0.01050220123592147};

  // compute coefficients for check
  const DataVector C = -(decay_time * d2tf_t0 + dtf_t0);
  const DataVector B = decay_time * (C - dtf_t0);
  const DataVector A = f_t0 - B;
  const std::array<DataVector, 3> init_func{{f_t0, dtf_t0, d2tf_t0}};

  INFO("Test with base class construction.");
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<domain::FunctionsOfTime::SettleToConstantQuaternion>(
          init_func, match_time, decay_time);
  test(f_of_t, match_time, decay_time, f_t0, dtf_t0, d2tf_t0, A);

  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t2 =
      serialize_and_deserialize(f_of_t);
  test(f_of_t2, match_time, decay_time, f_t0, dtf_t0, d2tf_t0, A);

  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t3 =
      f_of_t->get_clone();
  test(f_of_t3, match_time, decay_time, f_t0, dtf_t0, d2tf_t0, A);

  {
    INFO("Test operator==");
    CHECK(SettleToConstantQuaternion{init_func, match_time, decay_time} ==
          SettleToConstantQuaternion{init_func, match_time, decay_time});
    CHECK_FALSE(SettleToConstantQuaternion{init_func, match_time, decay_time} !=
                SettleToConstantQuaternion{init_func, match_time, decay_time});

    CHECK(SettleToConstantQuaternion{init_func, match_time, decay_time} !=
          SettleToConstantQuaternion{init_func, match_time, 2.0 * decay_time});
    CHECK_FALSE(
        SettleToConstantQuaternion{init_func, match_time, decay_time} ==
        SettleToConstantQuaternion{init_func, match_time, 2.0 * decay_time});

    CHECK(SettleToConstantQuaternion{init_func, match_time, decay_time} !=
          SettleToConstantQuaternion{init_func, 2.0 * match_time, decay_time});
    CHECK_FALSE(
        SettleToConstantQuaternion{init_func, match_time, decay_time} ==
        SettleToConstantQuaternion{init_func, 2.0 * match_time, decay_time});

    // Test == for different initial quaternions; both must be unit
    // quaternions.
    const DataVector another_f_t0{0.9928086358538663, 0.10830981865238282, 0.0,
                                  0.050990153534512375};
    const DataVector another_dtf_t0{-0.008379854510224357, 0.062163306368871823,
                                    0.0, 0.031117684009928405};
    const DataVector another_d2tf_t0{
        -0.008096991912484768, 0.022871837603575258, 0.0, 0.01291837713635158};
    const std::array<DataVector, 3> another_init_func{
        {another_f_t0, another_dtf_t0, another_d2tf_t0}};
    CHECK(
        SettleToConstantQuaternion{init_func, match_time, decay_time} !=
        SettleToConstantQuaternion{another_init_func, match_time, decay_time});
    CHECK_FALSE(
        SettleToConstantQuaternion{init_func, match_time, decay_time} ==
        SettleToConstantQuaternion{another_init_func, match_time, decay_time});

    CHECK(SettleToConstantQuaternion{init_func, match_time, decay_time} !=
          SettleToConstantQuaternion{
              {{init_func[0], init_func[1] + 1.0, init_func[2]}},
              match_time,
              decay_time});
    CHECK_FALSE(SettleToConstantQuaternion{init_func, match_time, decay_time} ==
                SettleToConstantQuaternion{
                    {{init_func[0], init_func[1] + 1.0, init_func[2]}},
                    match_time,
                    decay_time});

    CHECK(SettleToConstantQuaternion{init_func, match_time, decay_time} !=
          SettleToConstantQuaternion{
              {{init_func[0], init_func[1], init_func[2] + 1.0}},
              match_time,
              decay_time});
    CHECK_FALSE(SettleToConstantQuaternion{init_func, match_time, decay_time} ==
                SettleToConstantQuaternion{
                    {{init_func[0], init_func[1], init_func[2] + 1.0}},
                    match_time,
                    decay_time});
  }

  CHECK_THROWS_WITH((domain::FunctionsOfTime::SettleToConstantQuaternion{
                        {{DataVector{1.0, 0.0, 0.0, 0.0, 0.0},
                          DataVector{1.0, 0.0, 0.0, 0.0, 0.0},
                          DataVector{1.0, 0.0, 0.0, 0.0, 0.0}}},
                        10.0,
                        1.0}
                         .update(1.0, DataVector{}, 2.0)),
                    Catch::Matchers::ContainsSubstring(
                        "stored as DataVectors of size 4, not"));
  CHECK_THROWS_WITH(
      (domain::FunctionsOfTime::SettleToConstantQuaternion{
          {{DataVector{4.0, 0.0, 0.0, 0.0}, DataVector{1.0, 0.0, 0.0, 0.0},
            DataVector{1.0, 0.0, 0.0, 0.0}}},
          10.0,
          1.0}
           .update(1.0, DataVector{}, 2.0)),
      Catch::Matchers::ContainsSubstring(
          "should be a quaternion with a norm of 1.0, not"));
  CHECK_THROWS_WITH(
      (domain::FunctionsOfTime::SettleToConstantQuaternion{
          {{DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{1.0, 0.0, 0.0, 0.0},
            DataVector{1.0, 0.0, 0.0, 0.0}}},
          10.0,
          1.0}
           .update(1.0, DataVector{}, 2.0)),
      Catch::Matchers::ContainsSubstring("Cannot update this FunctionOfTime."));
}
