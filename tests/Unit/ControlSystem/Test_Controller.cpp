// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "ControlSystem/Controller.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
template <size_t DerivOrder>
void test_controller() {
  INFO("Test controller");
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const double initial_timescale = 1.0e-2;

  TimescaleTuner tst(std::vector<double>{initial_timescale}, max_timescale,
                     min_timescale, decrease_timescale_threshold,
                     increase_timescale_threshold, increase_factor,
                     decrease_factor);

  // test following a sinusoidal target function
  double t = 0.1;
  const double dt = 1.0e-3;
  const double final_time = 5.0;
  const double freq = 3.0;

  auto init_func = make_array<DerivOrder + 1, DataVector>(DataVector{1, 0.0});
  init_func[0] = {std::sin(freq * t)};
  init_func[1] = {freq * std::cos(freq * t)};
  init_func[2] = {-square(freq) * std::sin(freq * t)};
  if constexpr (DerivOrder > 2) {
    init_func[3] = {-cube(freq) * std::cos(freq * t)};
  }

  // properly initialize the function of time to match our target function
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<
          domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>>(
          t, init_func, t + dt);

  Controller<DerivOrder> control_signal{0.5};
  const double t_offset = 0.0;

  while (t < final_time) {
    auto target_func = make_array<DerivOrder, DataVector>(DataVector{1, 0.0});
    target_func[0] = {std::sin(freq * t)};
    target_func[1] = {freq * std::cos(freq * t)};
    auto lambda = target_func;
    if constexpr (DerivOrder == 2) {
      lambda = f_of_t->func_and_deriv(t);
    } else {
      target_func[2] = {-square(freq) * std::sin(freq * t)};
      lambda = f_of_t->func_and_2_derivs(t);
    }
    // check that the error is within the specified tolerance, which is
    // maintained by the TimescaleTuner adjusting the damping time
    CHECK(fabs(target_func[0][0] - lambda[0][0]) <
          decrease_timescale_threshold);

    // compute q = target - lambda
    // explicitly computing the derivatives of q here for testing purposes,
    // whereas, in practice, these will be computed numerically
    const auto q_and_derivs = target_func - lambda;

    // get the control signal for updating the FunctionOfTime
    const DataVector U = control_signal(t, tst.current_timescale(),
                                        q_and_derivs, t_offset, t_offset);

    t += dt;
    f_of_t->update(t, {U}, t + dt);

    // update the timescale
    tst.update_timescale({{q_and_derivs[0], q_and_derivs[1]}});
  }
}

template <size_t DerivOrder>
void test_timeoffsets() {
  INFO("Test time offsets");
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const double initial_timescale = 1.0e-2;

  TimescaleTuner tst(std::vector<double>{initial_timescale}, max_timescale,
                     min_timescale, decrease_timescale_threshold,
                     increase_timescale_threshold, increase_factor,
                     decrease_factor);

  // test following a sinusoidal target function
  double t = 0.1;
  const double dt = 1.0e-3;
  const double final_time = 5.0;
  const double freq = 3.0;

  // some vars for a rough averaging procedure
  const double alpha = 0.1;
  auto avg_qs = make_array<DerivOrder, DataVector>(DataVector{1, 0.0});
  double avg_time = 0.0;

  auto init_func = make_array<DerivOrder + 1, DataVector>(DataVector{1, 0.0});
  init_func[0] = {std::sin(freq * t)};
  init_func[1] = {freq * std::cos(freq * t)};
  init_func[2] = {-square(freq) * std::sin(freq * t)};
  if constexpr (DerivOrder > 2) {
    init_func[3] = {-cube(freq) * std::cos(freq * t)};
  }

  // properly initialize the function of time to match our target function
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<
          domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>>(
          t, init_func, t + dt);

  Controller<DerivOrder> control_signal{0.5};

  while (t < final_time) {
    auto target_func = make_array<DerivOrder, DataVector>(DataVector{1, 0.0});
    target_func[0] = {std::sin(freq * t)};
    target_func[1] = {freq * std::cos(freq * t)};
    auto lambda = target_func;
    if constexpr (DerivOrder == 2) {
      lambda = f_of_t->func_and_deriv(t);
    } else {
      target_func[2] = {-square(freq) * std::sin(freq * t)};
      lambda = f_of_t->func_and_2_derivs(t);
    }
    // check that the error is within the specified tolerance, which is
    // maintained by the TimescaleTuner adjusting the damping time
    CHECK(fabs(target_func[0][0] - lambda[0][0]) <
          decrease_timescale_threshold);

    // compute q = target - lambda
    // explicitly computing the derivatives of q here for testing purposes,
    // whereas, in practice, these will be computed numerically
    const auto q_and_derivs = target_func - lambda;

    // average q and its derivatives
    if (t == 0.1) {
      avg_time = t;
      avg_qs = q_and_derivs;
    } else {
      avg_time = alpha * t + (1.0 - alpha) * avg_time;
      avg_qs[0] = alpha * q_and_derivs[0] + (1.0 - alpha) * avg_qs[0];
      avg_qs[1] = alpha * q_and_derivs[1] + (1.0 - alpha) * avg_qs[1];
      if constexpr (DerivOrder > 2) {
        avg_qs[2] = alpha * q_and_derivs[2] + (1.0 - alpha) * avg_qs[2];
      }
    }

    // get the time offset due to averaging
    const double t_offset = t - avg_time;

    // get the control signal for updating the FunctionOfTime
    const DataVector U =
        control_signal(t, tst.current_timescale(), avg_qs, t_offset, t_offset);

    t += dt;
    f_of_t->update(t, {U}, t + dt);

    // update the timescale
    tst.update_timescale({{avg_qs[0], avg_qs[1]}});
  }
}

template <size_t DerivOrder>
void test_timeoffsets_noaverageq() {
  INFO("Test time offsets not averaging Q");
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;
  const double initial_timescale = 1.0e-2;

  TimescaleTuner tst(std::vector<double>{initial_timescale}, max_timescale,
                     min_timescale, decrease_timescale_threshold,
                     increase_timescale_threshold, increase_factor,
                     decrease_factor);

  // test following a sinusoidal target function
  double t = 0.1;
  const double dt = 1.0e-3;
  const double final_time = 5.0;
  const double freq = 3.0;

  // some vars for a rough averaging procedure
  const double alpha = 0.1;
  auto avg_qs = make_array<DerivOrder, DataVector>(DataVector{1, 0.0});
  double avg_time = 0.0;

  auto init_func = make_array<DerivOrder + 1, DataVector>(DataVector{1, 0.0});
  init_func[0] = {std::sin(freq * t)};
  init_func[1] = {freq * std::cos(freq * t)};
  init_func[2] = {-square(freq) * std::sin(freq * t)};
  if constexpr (DerivOrder > 2) {
    init_func[3] = {-cube(freq) * std::cos(freq * t)};
  }

  // properly initialize the function of time to match our target function
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> f_of_t =
      std::make_unique<
          domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>>(
          t, init_func, t + dt);

  Controller<DerivOrder> control_signal{0.5};

  while (t < final_time) {
    auto target_func = make_array<DerivOrder, DataVector>(DataVector{1, 0.0});
    target_func[0] = {std::sin(freq * t)};
    target_func[1] = {freq * std::cos(freq * t)};
    auto lambda = target_func;
    if constexpr (DerivOrder == 2) {
      lambda = f_of_t->func_and_deriv(t);
    } else {
      target_func[2] = {-square(freq) * std::sin(freq * t)};
      lambda = f_of_t->func_and_2_derivs(t);
    }
    // check that the error is within the specified tolerance, which is
    // maintained by the TimescaleTuner adjusting the damping time
    CHECK(fabs(target_func[0][0] - lambda[0][0]) <
          decrease_timescale_threshold);

    // compute q = target - lambda
    // explicitly computing the derivatives of q here for testing purposes,
    // whereas, in practice, these will be computed numerically
    const auto q_and_derivs = target_func - lambda;

    // average the derivatives of q (do not average q)
    if (t == 0.1) {
      avg_time = t;
      avg_qs = q_and_derivs;
    } else {
      avg_time = alpha * t + (1.0 - alpha) * avg_time;
      avg_qs[0] = q_and_derivs[0];
      avg_qs[1] = alpha * q_and_derivs[1] + (1.0 - alpha) * avg_qs[1];
      if constexpr (DerivOrder > 2) {
        avg_qs[2] = alpha * q_and_derivs[2] + (1.0 - alpha) * avg_qs[2];
      }
    }

    // since q is not averaged, there is no time offset
    const double q_t_offset = 0.0;
    // get the derivative time offset due to averaging
    const double t_offset = t - avg_time;

    // get the control signal for updating the FunctionOfTime
    const DataVector U = control_signal(t, tst.current_timescale(), avg_qs,
                                        q_t_offset, t_offset);

    t += dt;
    f_of_t->update(t, {U}, t + dt);

    // update the timescale
    tst.update_timescale({{avg_qs[0], avg_qs[1]}});
  }
}

template <size_t DerivOrder>
void test_equality_and_serialization() {
  INFO("Test equality and serialization");
  Controller<DerivOrder> controller1{0.5};
  Controller<DerivOrder> controller2{1.0};
  Controller<DerivOrder> controller3{0.5};
  controller3.assign_time_between_updates(2.0);

  CHECK(controller1 != controller2);
  CHECK_FALSE(controller1 == controller2);
  CHECK(controller1 != controller3);

  controller1.assign_time_between_updates(2.0);

  CHECK(controller1 == controller3);
  CHECK(controller1.get_update_fraction() == 0.5);

  Controller<DerivOrder> controller1_serialized =
      serialize_and_deserialize(controller1);

  CHECK(controller1 == controller1_serialized);
}

template <size_t DerivOrder>
void test_is_ready() {
  INFO("Test is ready");
  double time = 0.1;
  double min_timescale = 0.2;
  const double curr_expr_time = 0.2;
  auto avg_qs = make_array<DerivOrder, DataVector>(DataVector{1, 0.0});

  Controller<DerivOrder> controller{0.5};
  controller.set_initial_update_time(0.1);
  controller.assign_time_between_updates(min_timescale);

  CHECK_FALSE(controller.is_ready(time));
  time = curr_expr_time;
  CHECK(controller.is_ready(time));

  const double new_expr_time = controller.next_expiration_time(curr_expr_time);

  // Don't care what the control signal is, just that last_updated_time_ is
  // updated;
  const DataVector control_signal =
      controller(time, {min_timescale}, avg_qs, 0.0, 0.0);

  CHECK(controller.is_ready(new_expr_time));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Controller", "[ControlSystem][Unit]") {
  {
    INFO("DerivOrder 2");
    test_controller<2>();
    test_timeoffsets<2>();
    test_timeoffsets_noaverageq<2>();
    test_equality_and_serialization<2>();
    test_is_ready<2>();
  }
  {
    INFO("DerivOrder 3");
    test_controller<3>();
    test_timeoffsets<3>();
    test_timeoffsets_noaverageq<3>();
    test_equality_and_serialization<3>();
    test_is_ready<3>();
  }
}
