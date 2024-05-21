// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Time/TimeSteppers/ImexHelpers.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>

#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"

namespace TimeStepperTestUtils::imex {
void check_convergence_order(const ImexTimeStepper& stepper,
                             const std::pair<int32_t, int32_t>& step_range,
                             const bool output) {
  const auto do_integral = [&stepper](const int32_t num_steps) {
    const Slab slab(0., 1.);
    const TimeDelta step_size = slab.duration() / num_steps;

    TimeStepId time_step_id(true, 0, slab.start());
    double y = 1.;
    TimeSteppers::History<double> history{stepper.order()};
    TimeSteppers::History<double> implicit_history{stepper.order()};
    const auto rhs = [](const double v, const double /*t*/) { return 3.0 * v; };
    const auto implicit_rhs = [](const double v, const double /*t*/) {
      return -2.0 * v;
    };
    const auto implicit_rhs_init =
        [&implicit_rhs](const auto /*no_value*/, const double t) {
          return implicit_rhs(exp(t), t);
        };
    initialize_history(
        time_step_id.step_time(), make_not_null(&history),
        [](const double t) { return exp(t); }, rhs, step_size,
        stepper.number_of_past_steps());
    initialize_history(
        time_step_id.step_time(), make_not_null(&implicit_history),
        [](const double /*t*/) {
          return TimeSteppers::History<double>::no_value;
        },
        implicit_rhs_init, step_size, stepper.number_of_past_steps());
    while (time_step_id.step_time() < slab.end()) {
      history.insert(time_step_id, y, rhs(y, time_step_id.substep_time()));
      implicit_history.insert(time_step_id,
                              TimeSteppers::History<double>::no_value,
                              implicit_rhs(y, time_step_id.substep_time()));
      stepper.update_u(make_not_null(&y), history, step_size);
      stepper.clean_history(make_not_null(&history));
      // This system is simple enough that we can do the implicit
      // solve analytically.

      // Verify that the functions can be called in either order.  The
      // order used by the IMEX code has not been consistent during
      // development, so make sure to support both orders.
      auto y2 = y;
      auto implicit_history2 = implicit_history;
      stepper.add_inhomogeneous_implicit_terms(
          make_not_null(&y2), make_not_null(&implicit_history2), step_size);
      const double weight =
          stepper.implicit_weight(make_not_null(&implicit_history), step_size);
      // Both methods are required to do history cleanup
      CHECK(implicit_history == implicit_history2);
      // Verify that the weight calculation only uses the history times.
      implicit_history2.map_entries([](const auto value) {
        *value = std::numeric_limits<double>::signaling_NaN();
      });
      CHECK(stepper.implicit_weight(make_not_null(&implicit_history2),
                                    step_size) == weight);
      stepper.add_inhomogeneous_implicit_terms(
          make_not_null(&y), make_not_null(&implicit_history), step_size);
      CHECK(y == y2);

      y /= 1.0 + 2.0 * weight;
      time_step_id = stepper.next_time_id(time_step_id, step_size);
    }
    const double result = abs(y - exp(1.));
    return result;
  };
  CHECK(convergence_rate(step_range, 1, do_integral, output) ==
        approx(stepper.imex_order()).margin(0.4));
}

void check_bounded_dense_output(const ImexTimeStepper& stepper) {
  const double decay_constant = 1.0e5;
  const Slab slab(0.0, 1.0);
  const auto time_step = slab.duration();
  const auto rhs = [&](const double v, const double /*t*/) {
    return -decay_constant * v;
  };

  TimeStepId time_step_id(true, 0, slab.start());
  double y = 1.0;
  TimeSteppers::History<double> history{stepper.order()};
  initialize_history(
      time_step_id.step_time(), make_not_null(&history),
      [&](const double t) { return exp(-decay_constant * t); }, rhs, time_step,
      stepper.number_of_past_steps());
  double test_time = 0.0;
  for (;;) {
    history.insert(time_step_id, TimeSteppers::History<double>::no_value,
                   rhs(y, time_step_id.substep_time()));

    while (test_time < 1.0) {
      y = 1.0;
      if (not stepper.dense_update_u(make_not_null(&y), history, test_time)) {
        break;
      }
      CHECK(abs(y) < 5.0);
      test_time += 0.1;
    }

    if (time_step_id.slab_number() != 0) {
      break;
    }

    y = 1.0;
    stepper.add_inhomogeneous_implicit_terms(
        make_not_null(&y), make_not_null(&history), time_step);
    const double weight =
        stepper.implicit_weight(make_not_null(&history), time_step);

    y /= 1.0 + decay_constant * weight;
    time_step_id = stepper.next_time_id(time_step_id, time_step);
  }
  CHECK(test_time >= 1.0);
}
}  // namespace TimeStepperTestUtils::imex
