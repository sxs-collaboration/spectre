// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <type_traits>

#include "ErrorHandling/Assert.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeStepperTestUtils {

namespace {
template <typename F>
void take_step(
    const gsl::not_null<Time*> time,
    const gsl::not_null<double*> y,
    const gsl::not_null<TimeSteppers::History<double, double>*> history,
    const TimeStepper& stepper,
    F&& rhs,
    const TimeDelta& step_size) noexcept {
  TimeId time_id(step_size.is_positive(), 0, *time);
  for (uint64_t substep = 0;
       substep < stepper.number_of_substeps();
       ++substep) {
    CHECK(time_id.substep() == substep);
    history->insert(time_id.time(), *y, rhs(*y));
    stepper.update_u(y, history, step_size);
    time_id = stepper.next_time_id(time_id, step_size);
  }
  CHECK(time_id.time() - *time == step_size);
  *time = time_id.time();
}

template <typename F1, typename F2>
void initialize_history(
    Time time,
    const gsl::not_null<TimeSteppers::History<double, double>*> history,
    F1&& analytic,
    F2&& rhs,
    TimeDelta step_size,
    const size_t number_of_past_steps) noexcept {
  for (size_t j = 0; j < number_of_past_steps; ++j) {
    ASSERT(time.slab() == step_size.slab(), "Slab mismatch");
    if ((step_size.is_positive() and time.is_at_slab_start()) or
        (not step_size.is_positive() and time.is_at_slab_end())) {
      const Slab new_slab = time.slab().advance_towards(-step_size);
      time = time.with_slab(new_slab);
      step_size = step_size.with_slab(new_slab);
    }
    time -= step_size;
    history->insert_initial(time, analytic(time.value()),
                            rhs(analytic(time.value())));
  }
}
}  // namespace

void integrate_test(const TimeStepper& stepper,
                    const size_t number_of_past_steps,
                    const double integration_time,
                    const double epsilon) noexcept {
  auto analytic = [](double t) { return sin(t); };
  auto rhs = [](double v) { return sqrt(1. - square(v)); };

  const uint64_t num_steps = 800;
  const Slab slab = integration_time > 0
      ? Slab::with_duration_from_start(0., integration_time)
      : Slab::with_duration_to_end(0., -integration_time);
  const TimeDelta step_size = integration_time > 0
      ? slab.duration() / num_steps
      : -slab.duration() / num_steps;

  Time time = integration_time > 0 ? slab.start() : slab.end();
  double y = analytic(time.value());
  TimeSteppers::History<double, double> history;

  initialize_history(time, &history, analytic, rhs, step_size,
                     number_of_past_steps);

  for (uint64_t i = 0; i < num_steps; ++i) {
    take_step(&time, &y, &history, stepper, rhs, step_size);
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(history.size() < 20);
}

void integrate_variable_test(const TimeStepper& stepper,
                             const size_t number_of_past_steps,
                             const double epsilon) noexcept {
  auto analytic = [](double t) { return sin(t); };
  auto rhs = [](double v) { return sqrt(1. - square(v)); };

  const uint64_t num_steps = 800;
  const double average_step = 1. / num_steps;

  Slab slab = Slab::with_duration_to_end(0., average_step);
  Time time = slab.end();
  double y = analytic(time.value());

  TimeSteppers::History<double, double> history;
  initialize_history(time, &history, analytic, rhs, slab.duration(),
                     number_of_past_steps);

  for (uint64_t i = 0; i < num_steps; ++i) {
    slab = slab.advance().with_duration_from_start(
        (1. + 0.5 * sin(i)) * average_step);
    time = time.with_slab(slab);

    take_step(&time, &y, &history, stepper, rhs, slab.duration());
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
}

void stability_test(const TimeStepper& stepper) noexcept {
  const uint64_t num_steps = 5000;
  const double bracket_size = 1.1;

  // This is integrating dy/dt = -2y, which is chosen so that the stable
  // step size for Euler's method is 1.

  // Stable region
  {
    const Slab slab = Slab::with_duration_from_start(
        0., num_steps * stepper.stable_step() / bracket_size);
    const TimeDelta step_size = slab.duration() / num_steps;

    Time time = slab.start();
    double y = 1.;
    TimeSteppers::History<double, double> history;
    initialize_history(time, &history,
                       [](double t) { return exp(-2. * t); },
                       [](double v) { return -2. * v; },
                       step_size, stepper.number_of_past_steps());

    for (uint64_t i = 0; i < num_steps; ++i) {
      take_step(&time, &y, &history, stepper, [](double v) { return -2. * v; },
                step_size);
      CHECK(std::abs(y) < 10.);
    }
  }

  // Unstable region
  {
    const Slab slab = Slab::with_duration_from_start(
        0., num_steps * stepper.stable_step() * bracket_size);
    const TimeDelta step_size = slab.duration() / num_steps;

    Time time = slab.start();
    double y = 1.;
    TimeSteppers::History<double, double> history;
    initialize_history(time, &history,
                       [](double t) { return exp(-2. * t); },
                       [](double v) { return -2. * v; },
                       step_size, stepper.number_of_past_steps());

    for (uint64_t i = 0; i < num_steps; ++i) {
      take_step(&time, &y, &history, stepper, [](double v) { return -2. * v; },
                step_size);
      if (std::abs(y) > 10.) {
        return;
      }
    }
    CHECK(false);
  }
}

void equal_rate_boundary(const LtsTimeStepper& stepper,
                         const size_t number_of_past_steps,
                         const double epsilon, const bool forward) noexcept {
  // This does an integral putting the entire derivative into the
  // boundary term.
  const double unused_local_deriv = 4444.;

  auto analytic = [](double t) { return sin(t); };
  auto driver = [](double t) { return cos(t); };
  auto coupling = [=](const double& local, const double& remote) {
    CHECK(local == unused_local_deriv);
    return remote;
  };

  Approx approx = Approx::custom().epsilon(epsilon);

  const uint64_t num_steps = 100;
  const Slab slab(0.875, 1.);
  const TimeDelta step_size = (forward ? 1 : -1) * slab.duration() / num_steps;

  TimeId time_id(forward, 0, forward ? slab.start() : slab.end());
  double y = analytic(time_id.time().value());
  TimeSteppers::History<double, double> volume_history;
  TimeSteppers::BoundaryHistory<double, double, double> boundary_history;

  {
    Time history_time = time_id.time();
    TimeDelta history_step_size = step_size;
    for (size_t j = 0; j < number_of_past_steps; ++j) {
      ASSERT(history_time.slab() == history_step_size.slab(), "Slab mismatch");
      if ((history_step_size.is_positive() and
           history_time.is_at_slab_start()) or
          (not history_step_size.is_positive() and
           history_time.is_at_slab_end())) {
        const Slab new_slab =
            history_time.slab().advance_towards(-history_step_size);
        history_time = history_time.with_slab(new_slab);
        history_step_size = history_step_size.with_slab(new_slab);
      }
      history_time -= history_step_size;
      volume_history.insert_initial(history_time,
                                    analytic(history_time.value()), 0.);
      boundary_history.local_insert_initial(TimeId(forward, 0, history_time),
                                            unused_local_deriv);
      boundary_history.remote_insert_initial(TimeId(forward, 0, history_time),
                                             driver(history_time.value()));
    }
  }

  for (uint64_t i = 0; i < num_steps; ++i) {
    for (uint64_t substep = 0;
         substep < stepper.number_of_substeps();
         ++substep) {
      volume_history.insert(time_id.time(), y, 0.);
      boundary_history.local_insert(time_id, unused_local_deriv);
      boundary_history.remote_insert(time_id, driver(time_id.time().value()));

      stepper.update_u(make_not_null(&y), make_not_null(&volume_history),
                       step_size);
      y += stepper.compute_boundary_delta(
          coupling, make_not_null(&boundary_history), step_size);
      time_id = stepper.next_time_id(time_id, step_size);
    }
    CHECK(y == approx(analytic(time_id.time().value())));
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(boundary_history.local_size() < 20);
  CHECK(boundary_history.remote_size() < 20);
}

void check_convergence_order(const TimeStepper& stepper,
                             const int expected_order) noexcept {
  const auto do_integral = [&stepper](const int32_t num_steps) noexcept {
    const Slab slab(0., 1.);
    const TimeDelta step_size = slab.duration() / num_steps;

    Time time = slab.start();
    double y = 1.;
    TimeSteppers::History<double, double> history;
    initialize_history(time, &history, [](double t) { return exp(t); },
                       [](double v) { return v; }, step_size,
                       stepper.number_of_past_steps());
    while (time < slab.end()) {
      take_step(&time, &y, &history, stepper, [](double v) { return v; },
                step_size);
    }
    return abs(y - exp(1.));
  };
  const int32_t large_steps = 10;
  // The high-order solvers have round-off error around here
  const int32_t small_steps = 40;
  CHECK((log(do_integral(large_steps)) - log(do_integral(small_steps))) /
            (log(small_steps) - log(large_steps)) ==
        approx(expected_order).margin(0.4));
}

void check_dense_output(const TimeStepper& stepper,
                        const int expected_order) noexcept {
  const auto get_dense = [&stepper](TimeDelta step_size,
                                    const double time) noexcept {
    TimeId time_id(true, 0, step_size.slab().start());
    double y = 1.;
    TimeSteppers::History<double, double> history;
    initialize_history(time_id.time(), &history,
                       [](double t) { return exp(t); },
                       [](double v) { return v; }, step_size,
                       stepper.number_of_past_steps());
    for (;;) {
      // Dense output is done after the last substep
      const auto next_time_id = stepper.next_time_id(time_id, step_size);
      history.insert(time_id.time(), y, static_cast<double>(y));
      if (next_time_id.substep() == 0 and time < next_time_id.time().value()) {
        stepper.dense_update_u(make_not_null(&y), history, time);
        return y;
      }
      stepper.update_u(make_not_null(&y), make_not_null(&history), step_size);
      time_id = next_time_id;
      step_size = step_size.with_slab(time_id.time().slab());
    }
  };

  // Check that the dense output is continuous
  {
    const Slab slab(0., 1.);
    Time time = slab.start();
    double y = 1.;
    TimeSteppers::History<double, double> history;
    initialize_history(time, &history, [](double t) { return exp(t); },
                       [](double v) { return v; }, slab.duration(),
                       stepper.number_of_past_steps());
    take_step(&time, &y, &history, stepper, [](double v) { return v; },
              slab.duration());

    CHECK(get_dense(slab.duration(), std::numeric_limits<double>::epsilon()) ==
          approx(1.));
    CHECK(get_dense(slab.duration(),
                    1. - std::numeric_limits<double>::epsilon()) == approx(y));
  }

  // Test convergence
  {
    const int32_t large_steps = 10;
    // The high-order solvers have round-off error around here
    const int32_t small_steps = 40;

    const auto error = [&get_dense](const int32_t steps) noexcept {
      const Slab slab(0., 1.);
      return abs(get_dense(slab.duration() / steps, 0.25 * M_PI) -
                 exp(0.25 * M_PI));
    };
    CHECK((log(error(large_steps)) - log(error(small_steps))) /
              (log(small_steps) - log(large_steps)) ==
          approx(expected_order).margin(0.4));
  }
}

void check_boundary_dense_output(const LtsTimeStepper& stepper) noexcept {
  // We only support variable time-step, multistep LTS integration.
  // Any multistep, variable time-step integrator must give the same
  // results from dense output as from just taking a short step
  // because we require dense output to be continuous.  A sufficient
  // test is therefore to run with an LTS pattern and check that the
  // dense output predicts the actual step result.
  const Slab slab(0., 1.);

  // We don't use any meaningful values.  We only care that the dense
  // output gives the same result as normal output.
  auto get_value = [value = 1.]() mutable noexcept { return value *= 1.1; };

  const auto coupling = [](const double a, const double b) noexcept {
    return a * b;
  };

  const auto make_time_id = [](const Time& t) noexcept {
    return TimeId(true, 0, t);
  };

  TimeSteppers::BoundaryHistory<double, double, double> history;
  {
    const Slab init_slab = slab.retreat();
    for (size_t i = 0; i < stepper.number_of_past_steps(); ++i) {
      const Time init_time =
          init_slab.end() -
          init_slab.duration() * (i + 1) / stepper.number_of_past_steps();
      history.local_insert_initial(make_time_id(init_time), get_value());
      history.remote_insert_initial(make_time_id(init_time), get_value());
    }
  }

  std::array<std::deque<TimeDelta>, 2> dt{
      {{slab.duration() / 2, slab.duration() / 4, slab.duration() / 4},
       {slab.duration() / 6, slab.duration() / 6, slab.duration() * 2 / 9,
        slab.duration() * 4 / 9}}};

  Time t = slab.start();
  double y = 0.;
  Time next_check = t + dt[0][0];
  std::array<Time, 2> next{{t, t}};
  for (;;) {
    const auto side = static_cast<size_t>(
        std::min_element(next.cbegin(), next.cend()) - next.cbegin());

    if (side == 0) {
      history.local_insert(make_time_id(next[0]), get_value());
    } else {
      history.remote_insert(make_time_id(next[1]), get_value());
    }

    const TimeDelta this_dt = gsl::at(dt, side).front();
    gsl::at(dt, side).pop_front();

    gsl::at(next, side) += this_dt;

    if (*std::min_element(next.cbegin(), next.cend()) == next_check) {
      const double dense_result =
          stepper.boundary_dense_output(coupling, history, next_check.value());
      const double delta = stepper.compute_boundary_delta(
          coupling, make_not_null(&history), next_check - t);
      CHECK(dense_result == approx(delta));
      y += delta;
      if (next_check.is_at_slab_boundary()) {
        break;
      }
      t = next_check;
      next_check += dt[0].front();
    }
  }
}
}  // namespace TimeStepperTestUtils
