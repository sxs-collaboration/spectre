// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <catch.hpp>
#include <cmath>
#include <deque>
#include <tuple>

#include "ErrorHandling/Assert.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

template <typename Stepper, typename F>
void take_step(
    const gsl::not_null<Time*> time,
    const gsl::not_null<double*> y,
    const gsl::not_null<std::deque<std::tuple<Time, double, double>>*> history,
    const Stepper& stepper,
    F&& rhs,
    const TimeDelta& step_size) noexcept {
  TimeDelta accumulated_step_dt = step_size * 0;
  for (size_t substep = 0; substep < stepper.number_of_substeps(); ++substep) {
    history->emplace_back(*time, *y, rhs(*y));
    const TimeDelta substep_dt = stepper.update_u(y, *history, step_size);
    accumulated_step_dt += substep_dt;
    *time += substep_dt;
  }
  CHECK(accumulated_step_dt == step_size);
  while (history->size() > stepper.number_of_past_steps()) {
    history->pop_front();
  }
}

template <typename Stepper, typename F1, typename F2>
void initialize_history(
    Time time,
    const gsl::not_null<std::deque<std::tuple<Time, double, double>>*> history,
    const Stepper& stepper,
    F1&& analytic,
    F2&& rhs,
    TimeDelta step_size) noexcept {
  for (size_t j = 0; j < stepper.number_of_past_steps(); ++j) {
    ASSERT(time.slab() == step_size.slab(), "Slab mismatch");
    if ((step_size.is_positive() and time.is_at_slab_start()) or
        (not step_size.is_positive() and time.is_at_slab_end())) {
      const Slab new_slab = time.slab().advance_towards(-step_size);
      time = time.with_slab(new_slab);
      step_size = step_size.with_slab(new_slab);
    }
    time -= step_size;
    history->emplace_front(time, analytic(time.value()),
                           rhs(analytic(time.value())));
  }
}

template <typename Stepper>
void check_multistep_properties(const Stepper& stepper) noexcept {
  CHECK(stepper.number_of_substeps() == 1);
}

template <typename Stepper>
void check_substep_properties(const Stepper& stepper) noexcept {
  CHECK(stepper.is_self_starting());
  CHECK(stepper.number_of_past_steps() == 0);
}

template <typename Stepper>
void integrate_test(const Stepper& stepper, const double integration_time,
                    const double epsilon) noexcept {
  auto analytic = [](double t) { return sin(t); };
  auto rhs = [](double v) { return sqrt(1. - square(v)); };

  const size_t num_steps = 800;
  const Slab slab = integration_time > 0
      ? Slab::with_duration_from_start(0., integration_time)
      : Slab::with_duration_to_end(0., -integration_time);
  const TimeDelta step_size = integration_time > 0
      ? slab.duration() / num_steps
      : -slab.duration() / num_steps;

  Time time = integration_time > 0 ? slab.start() : slab.end();
  double y = analytic(time.value());
  std::deque<std::tuple<Time, double, double>> history;

  if (not stepper.is_self_starting()) {
    initialize_history(time, &history, stepper, analytic, rhs, step_size);
  }

  for (size_t i = 0; i < num_steps; ++i) {
    take_step(&time, &y, &history, stepper, rhs, step_size);
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
}

template <typename Stepper>
void integrate_variable_test(const Stepper& stepper,
                             const double epsilon) noexcept {
  auto analytic = [](double t) { return sin(t); };
  auto rhs = [](double v) { return sqrt(1. - square(v)); };

  const size_t num_steps = 800;
  const double average_step = 1. / num_steps;

  Slab slab = Slab::with_duration_to_end(0., average_step);
  Time time = slab.end();
  double y = analytic(time.value());

  std::deque<std::tuple<Time, double, double>> history;
  if (not stepper.is_self_starting()) {
    initialize_history(time, &history, stepper, analytic, rhs, slab.duration());
  }

  for (size_t i = 0; i < num_steps; ++i) {
    slab = slab.advance().with_duration_from_start(
        (1. + 0.5 * sin(i)) * average_step);

    take_step(&time, &y, &history, stepper, rhs, slab.duration());
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
}

template <typename Stepper>
void stability_test(const Stepper& stepper) noexcept {
  const size_t num_steps = 5000;
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
    std::deque<std::tuple<Time, double, double>> history;
    if (not stepper.is_self_starting()) {
      initialize_history(time, &history, stepper,
                         [](double t) { return exp(-2. * t); },
                         [](double v) { return -2. * v; },
                         step_size);
    }

    for (size_t i = 0; i < num_steps; ++i) {
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
    std::deque<std::tuple<Time, double, double>> history;
    if (not stepper.is_self_starting()) {
      initialize_history(time, &history, stepper,
                         [](double t) { return exp(-2. * t); },
                         [](double v) { return -2. * v; },
                         step_size);
    }

    for (size_t i = 0; i < num_steps; ++i) {
      take_step(&time, &y, &history, stepper, [](double v) { return -2. * v; },
                step_size);
      if (std::abs(y) > 10.) {
        return;
      }
    }
    CHECK(false);
  }
}
