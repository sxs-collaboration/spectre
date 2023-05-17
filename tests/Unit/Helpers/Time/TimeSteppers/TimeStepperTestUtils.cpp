// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DynamicMatrix.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"

namespace TimeStepperTestUtils {

namespace {
template <typename T, typename F>
void take_step(
    const gsl::not_null<Time*> time, const gsl::not_null<T*> y,
    const gsl::not_null<TimeSteppers::History<T>*> history,
    const TimeStepper& stepper, F&& rhs, const TimeDelta& step_size) {
  TimeStepId time_id(step_size.is_positive(), 0, *time);
  for (uint64_t substep = 0;
       substep < stepper.number_of_substeps();
       ++substep) {
    CHECK(time_id.substep() == substep);
    history->insert(time_id, *y, rhs(*y, time_id.substep_time()));
    // There is no std::numeric_limits<complex>
    *y = std::numeric_limits<double>::signaling_NaN();
    stepper.update_u(y, history, step_size);
    time_id = stepper.next_time_id(time_id, step_size);
  }
  CHECK(time_id.substep() == 0);
  CHECK(time_id.step_time() - *time == step_size);
  *time = time_id.step_time();
}

template <typename F>
void take_step_and_check_error(
    const gsl::not_null<Time*> time, const gsl::not_null<double*> y,
    const gsl::not_null<double*> y_error,
    const gsl::not_null<TimeSteppers::History<double>*> history,
    const TimeStepper& stepper, F&& rhs, const TimeDelta& step_size) {
  TimeStepId time_id(step_size.is_positive(), 0, *time);
  for (uint64_t substep = 0; substep < stepper.number_of_substeps_for_error();
       ++substep) {
    CHECK(time_id.substep() == substep);
    history->insert(time_id, *y, rhs(*y, time_id.substep_time()));
    *y = std::numeric_limits<double>::signaling_NaN();
    bool error_updated = stepper.update_u(y, y_error, history, step_size);
    CAPTURE(substep);
    REQUIRE((substep == stepper.number_of_substeps_for_error() - 1) ==
            error_updated);
    time_id = stepper.next_time_id_for_error(time_id, step_size);
  }
  CHECK(time_id.substep() == 0);
  CHECK(time_id.step_time() - *time == step_size);
  *time = time_id.step_time();
}

template <typename F>
double convergence_rate(const int32_t large_steps, const int32_t small_steps,
                        F&& error, const bool output = false) {
  // We do a least squares fit on a log-log error-vs-steps plot.  The
  // unequal points caused by the log scale will introduce some bias,
  // but the typical range this is used for is only a factor of a few,
  // so it shouldn't be too bad.

  // Make sure testing code is not left enabled.
  CHECK(not output);

  std::ofstream output_stream{};
  if (output) {
    output_stream.open("convergence.dat");
    output_stream.precision(18);
  }

  const auto num_tests = static_cast<size_t>(small_steps - large_steps) + 1;
  std::vector<double> log_steps;
  std::vector<double> log_errors;
  log_steps.reserve(num_tests);
  log_errors.reserve(num_tests);
  for (auto steps = large_steps; steps <= small_steps; ++steps) {
    const double this_error = abs(error(steps));
    if (output) {
      output_stream << steps << "\t" << this_error << std::endl;
    }
    log_steps.push_back(log(steps));
    log_errors.push_back(log(this_error));
  }
  const double average_log_steps = alg::accumulate(log_steps, 0.0) / num_tests;
  const double average_log_errors =
      alg::accumulate(log_errors, 0.0) / num_tests;
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < num_tests; ++i) {
    numerator += (log_steps[i] - average_log_steps) *
        (log_errors[i] - average_log_errors);
    denominator += square(log_steps[i] - average_log_steps);
  }
  return -numerator / denominator;
}
}  // namespace

void check_multistep_properties(const TimeStepper& stepper) {
  CHECK(stepper.number_of_substeps() == 1);
}

void check_substep_properties(const TimeStepper& stepper) {
  CHECK(stepper.number_of_past_steps() == 0);

  const Slab slab(0., 1.);
  TimeStepId id(true, 3, slab.start() + slab.duration() / 2);
  TimeSteppers::History<double> history{stepper.order()};
  CHECK(stepper.can_change_step_size(id, history));
  history.insert(id, 0.0, 0.0);
  id = stepper.next_time_id(id, slab.duration() / 2);
  if (id.substep() != 0) {
    CHECK(not stepper.can_change_step_size(id, history));
  }
}

void integrate_test(const TimeStepper& stepper, const size_t order,
                    const size_t number_of_past_steps,
                    const double integration_time, const double epsilon) {
  auto analytic = [](const double t) { return sin(t); };
  auto rhs = [](const double v, const double /*t*/) {
    return sqrt(1. - square(v));
  };

  const uint64_t num_steps = 800;
  const Slab slab = integration_time > 0
      ? Slab::with_duration_from_start(0., integration_time)
      : Slab::with_duration_to_end(0., -integration_time);
  const TimeDelta step_size = integration_time > 0
      ? slab.duration() / num_steps
      : -slab.duration() / num_steps;

  Time time = integration_time > 0 ? slab.start() : slab.end();
  double y = analytic(time.value());
  TimeSteppers::History<double> history{order};

  initialize_history(time, make_not_null(&history), analytic, rhs, step_size,
                     number_of_past_steps);

  for (uint64_t i = 0; i < num_steps; ++i) {
    take_step(&time, make_not_null(&y), make_not_null(&history), stepper, rhs,
              step_size);
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(history.size() < 20);
}

void integrate_test_explicit_time_dependence(const TimeStepper& stepper,
                                             const size_t order,
                                             const size_t number_of_past_steps,
                                             const double integration_time,
                                             const double epsilon) {
  auto analytic = [](const double t) { return square(t); };
  auto rhs = [](const double /*v*/, const double t) { return 2 * t; };

  const uint64_t num_steps = 800;
  const Slab slab = integration_time > 0
                        ? Slab::with_duration_from_start(0., integration_time)
                        : Slab::with_duration_to_end(0., -integration_time);
  const TimeDelta step_size = integration_time > 0
                                  ? slab.duration() / num_steps
                                  : -slab.duration() / num_steps;

  Time time = integration_time > 0 ? slab.start() : slab.end();
  double y = analytic(time.value());
  TimeSteppers::History<double> history{order};

  initialize_history(time, make_not_null(&history), analytic, rhs, step_size,
                     number_of_past_steps);

  for (uint64_t i = 0; i < num_steps; ++i) {
    take_step(&time, make_not_null(&y), make_not_null(&history), stepper, rhs,
              step_size);
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(history.size() < 20);
}

void integrate_error_test(const TimeStepper& stepper, const size_t order,
                          const size_t number_of_past_steps,
                          const double integration_time, const double epsilon,
                          const size_t num_steps, const double error_factor) {
  auto analytic = [](const double t) { return sin(t); };
  auto rhs = [](const double v, const double /*t*/) {
    return sqrt(1. - square(v));
  };

  const Slab slab = integration_time > 0
      ? Slab::with_duration_from_start(0., integration_time)
      : Slab::with_duration_to_end(0., -integration_time);
  const TimeDelta step_size = integration_time > 0
      ? slab.duration() / num_steps
      : -slab.duration() / num_steps;

  Time time = integration_time > 0 ? slab.start() : slab.end();
  double y = analytic(time.value());
  TimeSteppers::History<double> history{order};

  initialize_history(time, make_not_null(&history), analytic, rhs, step_size,
                     number_of_past_steps);
  double y_error = std::numeric_limits<double>::signaling_NaN();
  double previous_y = std::numeric_limits<double>::signaling_NaN();
  double previous_time = std::numeric_limits<double>::signaling_NaN();
  for (uint64_t i = 0; i < num_steps; ++i) {
    take_step_and_check_error(make_not_null(&time), make_not_null(&y),
                              make_not_null(&y_error), make_not_null(&history),
                              stepper, rhs, step_size);
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));

    // check that the error measure is a reasonable estimate of the deviation
    // from the analytic solution. This solution is smooth, so the error should
    // be dominated by the stepper.
    if (i > num_steps / 2) {
      double local_error = abs((y - analytic(time.value())) -
                               (previous_y - analytic(previous_time)));
      CHECK(local_error < std::max(abs(y_error), 1e-14));
      CHECK(local_error > abs(error_factor * y_error));
    }
    previous_y = y;
    previous_time = time.value();
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(history.size() < 20);
}

void integrate_variable_test(const TimeStepper& stepper, const size_t order,
                             const size_t number_of_past_steps,
                             const double epsilon) {
  auto analytic = [](const double t) { return sin(t); };
  auto rhs = [](const double v, const double /*t*/) {
    return sqrt(1. - square(v));
  };

  const uint64_t num_steps = 800;
  const double average_step = 1. / num_steps;

  Slab slab = Slab::with_duration_to_end(0., average_step);
  Time time = slab.end();
  double y = analytic(time.value());

  TimeSteppers::History<double> history{order};
  initialize_history(time, make_not_null(&history), analytic, rhs,
                     slab.duration(), number_of_past_steps);

  for (uint64_t i = 0; i < num_steps; ++i) {
    slab = slab.advance().with_duration_from_start(
        (1. + 0.5 * sin(i)) * average_step);
    time = time.with_slab(slab);

    take_step(&time, make_not_null(&y), make_not_null(&history), stepper, rhs,
              slab.duration());
    // This check needs a looser tolerance for lower-order time steppers.
    CHECK(y == approx(analytic(time.value())).epsilon(epsilon));
  }
}

namespace {
// This checks the stability of dy/dt = [exp(i phase) - 1] y.  To do
// this, it uses the fact that time steppers can be considered as
// linear operators on their history values.  It constructs the
// operator for the passed stepper with the given step and phase
// explicitly, and then checks whether all the eigenvalues have
// magnitude less than one.
bool step_is_stable(const TimeStepper& stepper, const double step_size,
                    const double phase) {
  const size_t operator_size = stepper.number_of_past_steps() + 1;
  const Slab slab(0.0, step_size * (operator_size + 1));
  const auto step_delta = slab.duration() / (operator_size + 1);
  const std::complex<double> coefficient = std::polar(1.0, phase) - 1.0;
  const auto rhs = [&coefficient](const std::complex<double> v,
                                  const double /*t*/) {
    return coefficient * v;
  };

  blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>
      stepper_operator(operator_size, operator_size, 0.0);
  // Add entries for shifting the history backwards after the step.
  for (size_t i = 0; i < operator_size - 1; ++i) {
    stepper_operator(i, i + 1) = 1.0;
  }

  // Construct the entries for the linear combination forming the new
  // value.
  for (size_t step_to_test = 0; step_to_test < operator_size; ++step_to_test) {
    TimeSteppers::History<std::complex<double>> history{stepper.order()};
    TimeStepId id(true, 0, slab.start());
    for (size_t past_step = 0; past_step < operator_size - 1; ++past_step) {
      if (past_step == step_to_test) {
        history.insert(id, 1.0, rhs(1.0, 0.0));
      } else {
        history.insert(id, 0.0, 0.0);
      }
      id = id.next_step(step_delta);
    }
    Time time = id.step_time();
    std::complex<double> y = step_to_test == operator_size - 1 ? 1.0 : 0.0;
    take_step(make_not_null(&time), make_not_null(&y), make_not_null(&history),
              stepper, rhs, step_delta);
    stepper_operator(operator_size - 1, step_to_test) = y;
  }

  const blaze::DynamicVector<std::complex<double>> stepper_eigenvalues =
      eigen(stepper_operator);
  return alg::all_of(stepper_eigenvalues, [](const std::complex<double> e) {
    return norm(e) <= 1.0;
  });
}
}  // namespace

void stability_test(const TimeStepper& stepper, const double phase) {
  const double bracket_size = 1.001;

  CHECK(step_is_stable(stepper, stepper.stable_step() / bracket_size, phase));
  CHECK(
      not step_is_stable(stepper, stepper.stable_step() * bracket_size, phase));

  // Check that the phase is correct.  Doing a global check is too
  // expensive, but we can check that the passed value is a local
  // minimum.
  CHECK(step_is_stable(stepper, stepper.stable_step() / bracket_size,
                       phase + 0.1));
  CHECK(step_is_stable(stepper, stepper.stable_step() / bracket_size,
                       phase - 0.1));
}

void equal_rate_boundary(const LtsTimeStepper& stepper, const size_t order,
                         const size_t number_of_past_steps,
                         const double epsilon, const bool forward) {
  // This does an integral putting the entire derivative into the
  // boundary term.
  const double unused_local_deriv = 4444.;

  auto analytic = [](double t) { return sin(t); };
  auto driver = [](double t) { return cos(t); };
  auto coupling = [=](const double local, const double remote) {
    CHECK(local == unused_local_deriv);
    return remote;
  };

  Approx approx = Approx::custom().epsilon(epsilon);

  const uint64_t num_steps = 100;
  const Slab slab(0.875, 1.);
  const TimeDelta step_size = (forward ? 1 : -1) * slab.duration() / num_steps;

  TimeStepId time_id(forward, 0, forward ? slab.start() : slab.end());
  double y = analytic(time_id.substep_time());
  TimeSteppers::History<double> volume_history{order};
  TimeSteppers::BoundaryHistory<double, double, double> boundary_history{order};

  {
    Time history_time = time_id.step_time();
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
      const TimeStepId history_id(forward, 0, history_time);
      volume_history.insert_initial(history_id, analytic(history_time.value()),
                                    0.);
      boundary_history.local_insert_initial(history_id, unused_local_deriv);
      boundary_history.remote_insert_initial(history_id,
                                             driver(history_time.value()));
    }
  }

  for (uint64_t i = 0; i < num_steps; ++i) {
    for (uint64_t substep = 0;
         substep < stepper.number_of_substeps();
         ++substep) {
      volume_history.insert(time_id, y, 0.);
      boundary_history.local_insert(time_id, unused_local_deriv);
      boundary_history.remote_insert(time_id, driver(time_id.substep_time()));

      stepper.update_u(make_not_null(&y), make_not_null(&volume_history),
                       step_size);
      stepper.add_boundary_delta(&y, make_not_null(&boundary_history),
                                 step_size, coupling);
      time_id = stepper.next_time_id(time_id, step_size);
    }
    CHECK(y == approx(analytic(time_id.substep_time())));
  }
  // Make sure history is being cleaned up.  The limit of 20 is
  // arbitrary, but much larger than the order of any integrators we
  // care about and much smaller than the number of time steps in the
  // test.
  CHECK(boundary_history.local_size() < 20);
  CHECK(boundary_history.remote_size() < 20);
}

void check_convergence_order(const TimeStepper& stepper,
                             const std::pair<int32_t, int32_t>& step_range,
                             const bool output) {
  const auto do_integral = [&stepper](const int32_t num_steps) {
    const Slab slab(0., 1.);
    const TimeDelta step_size = slab.duration() / num_steps;

    Time time = slab.start();
    double y = 1.;
    TimeSteppers::History<double> history{stepper.order()};
    const auto rhs = [](const double v, const double /*t*/) { return v; };
    initialize_history(
        time, make_not_null(&history), [](const double t) { return exp(t); },
        rhs, step_size, stepper.number_of_past_steps());
    while (time < slab.end()) {
      take_step(&time, make_not_null(&y), make_not_null(&history), stepper, rhs,
                step_size);
    }
    const double result = abs(y - exp(1.));
    return result;
  };
  CHECK(convergence_rate(step_range.first, step_range.second, do_integral,
                         output) == approx(stepper.order()).margin(0.4));
}

void check_dense_output(const TimeStepper& stepper,
                        const size_t history_integration_order) {
  const auto get_dense = [&stepper, &history_integration_order](
                             const TimeDelta& step_size, const double time) {
    const auto impl = [&stepper, &history_integration_order, &step_size,
                       &time](const bool use_error_methods) {
      CAPTURE(use_error_methods);
      TimeStepId time_id(step_size.is_positive(), 0,
                         step_size.is_positive() ? step_size.slab().start()
                                                 : step_size.slab().end());
      const evolution_less<double> before{time_id.time_runs_forward()};
      double y = 1.;
      TimeSteppers::History<double> history{history_integration_order};
      initialize_history(
          time_id.step_time(), make_not_null(&history),
          [](const double t) { return exp(t); },
          [](const double v, const double /*t*/) { return v; }, step_size,
          stepper.number_of_past_steps());
      auto step = step_size;
      for (;;) {
        history.insert(time_id, y, y);
        y = std::numeric_limits<double>::signaling_NaN();
        if (not before((time_id.step_time() + step).value(), time)) {
          if (stepper.dense_update_u(make_not_null(&y), history, time)) {
            return y;
          }
          REQUIRE(before(time_id.step_time().value(), time));
        }
        if (use_error_methods) {
          double error = 0.0;
          stepper.update_u(make_not_null(&y), make_not_null(&error),
                           make_not_null(&history), step);
        } else {
          stepper.update_u(make_not_null(&y), make_not_null(&history), step);
        }
        time_id = use_error_methods
                      ? stepper.next_time_id_for_error(time_id, step)
                      : stepper.next_time_id(time_id, step);
        step = step.with_slab(time_id.step_time().slab());
      }
    };
    const double with_error = impl(true);
    const double without_error = impl(false);
    CHECK(with_error == approx(without_error));
    return without_error;
  };

  // Check that the dense output is continuous
  {
    auto local_approx = approx.epsilon(1e-12);
    for (const auto time_step :
         {Slab(0., 1.).duration(), -Slab(-1., 0.).duration()}) {
      CAPTURE(time_step);
      Time time = Slab(0., 1.).start().with_slab(time_step.slab());
      double y = 1.;
      TimeSteppers::History<double> history{history_integration_order};
      const auto rhs = [](const double v, const double /*t*/) { return v; };
      initialize_history(
          time, make_not_null(&history), [](const double t) { return exp(t); },
          rhs, time_step, stepper.number_of_past_steps());
      take_step(&time, make_not_null(&y), make_not_null(&history), stepper, rhs,
                time_step);

      // Some time steppers special-case the endpoints of the
      // interval, so check just inside to trigger the main dense
      // output path.
      CHECK(get_dense(time_step, 0.0) == local_approx(1.));
      CHECK(get_dense(time_step, std::numeric_limits<double>::epsilon() *
                                     time_step.value()) == local_approx(1.));
      CHECK(get_dense(time_step, (1. - std::numeric_limits<double>::epsilon()) *
                                     time.value()) == local_approx(y));
      CHECK(get_dense(time_step, time.value()) == local_approx(y));
    }
  }

  // Test convergence
  {
    const int32_t large_steps = 10;
    // The high-order solvers have round-off error around here
    const int32_t small_steps = 30;

    const auto error = [&get_dense](const int32_t steps) {
      const Slab slab(0., 1.);
      return abs(get_dense(slab.duration() / steps, 0.25 * M_PI) -
                 exp(0.25 * M_PI));
    };
    CHECK(convergence_rate(large_steps, small_steps, error) ==
          approx(history_integration_order).margin(0.4));

    const auto error_backwards = [&get_dense](const int32_t steps) {
      const Slab slab(-1., 0.);
      return abs(get_dense(-slab.duration() / steps, -0.25 * M_PI) -
                 exp(-0.25 * M_PI));
    };
    CHECK(convergence_rate(large_steps, small_steps, error_backwards) ==
          approx(history_integration_order).margin(0.4));
  }
}

void check_boundary_dense_output(const LtsTimeStepper& stepper) {
  // We only support variable time-step, multistep LTS integration.
  // Any multistep, variable time-step integrator must give the same
  // results from dense output as from just taking a short step
  // because we require dense output to be continuous.  A sufficient
  // test is therefore to run with an LTS pattern and check that the
  // dense output predicts the actual step result.
  const Slab slab(0., 1.);

  // We don't use any meaningful values.  We only care that the dense
  // output gives the same result as normal output.
  // NOLINTNEXTLINE(spectre-mutable)
  auto get_value = [value = 1.]() mutable { return value *= 1.1; };

  const auto coupling = [](const double a, const double b) { return a * b; };

  const auto make_time_id = [](const Time& t) {
    return TimeStepId(true, 0, t);
  };

  TimeSteppers::BoundaryHistory<double, double, double> history{
    stepper.order()};
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
  Time next_check = t + dt[0][0];
  std::array<Time, 2> next{{t, t}};
  for (;;) {
    const size_t side = next[0] <= next[1] ? 0 : 1;

    if (side == 0) {
      history.local_insert(make_time_id(next[0]), get_value());
    } else {
      history.remote_insert(make_time_id(next[1]), get_value());
    }

    const TimeDelta this_dt = gsl::at(dt, side).front();
    gsl::at(dt, side).pop_front();

    gsl::at(next, side) += this_dt;

    if (std::min(next[0], next[1]) == next_check) {
      double dense_result = 0.0;
      stepper.boundary_dense_output(&dense_result, history, next_check.value(),
                                    coupling);
      double delta = 0.0;
      stepper.add_boundary_delta(&delta, make_not_null(&history),
                                 next_check - t, coupling);
      CHECK(dense_result == approx(delta));
      if (next_check.is_at_slab_boundary()) {
        break;
      }
      t = next_check;
      next_check += dt[0].front();
    }
  }
}

void check_strong_stability_preservation(const TimeStepper& stepper,
                                         const double step_size) {
  // Check that each substep is a convex combination of partial Euler steps.
  ASSERT(stepper.number_of_past_steps() == 0,
         "Unimplemented for multistep methods");

  const auto impl = [&step_size](const size_t order,
                                 const uint64_t number_of_substeps_to_test,
                                 const auto update_u, const auto next_time_id) {
    const Slab slab(0.0, step_size);
    const auto time_step = slab.duration();

    for (uint64_t substep_being_varied = 0;
         substep_being_varied < number_of_substeps_to_test;
         ++substep_being_varied) {
      CAPTURE(substep_being_varied);
      TimeStepId time_step_id(true, 0, slab.start());
      TimeSteppers::History<double> value_history(order);
      TimeSteppers::History<double> derivative_history(order);
      for (uint64_t substep_being_tested = 0;
           substep_being_tested < number_of_substeps_to_test;
           ++substep_being_tested) {
        CAPTURE(substep_being_tested);
        if (substep_being_tested == substep_being_varied) {
          value_history.insert(time_step_id, 1.0, 0.0);
          derivative_history.insert(time_step_id, 0.0, 1.0);
        } else {
          value_history.insert(time_step_id, 0.0, 0.0);
          derivative_history.insert(time_step_id, 0.0, 0.0);
        }
        double u = std::numeric_limits<double>::signaling_NaN();
        update_u(make_not_null(&u), make_not_null(&value_history), time_step);
        const double value_dependence = u;
        update_u(make_not_null(&u), make_not_null(&derivative_history),
                 time_step);
        const double derivative_dependence = u;
        CHECK(value_dependence >= 0.0);
        CHECK(derivative_dependence >= 0.0);
        CHECK(step_size * derivative_dependence <= approx(value_dependence));
        time_step_id = next_time_id(time_step_id, time_step);
      }
    }
  };

  {
    INFO("Without error estimate");
    impl(
        stepper.order(), stepper.number_of_substeps(),
        [&](const gsl::not_null<double*> u,
            const gsl::not_null<TimeSteppers::History<double>*> history,
            const TimeDelta& time_step) {
          stepper.update_u(u, history, time_step);
        },
        [&](const TimeStepId& time_step_id, const TimeDelta& time_step) {
          return stepper.next_time_id(time_step_id, time_step);
        });
  }
  {
    INFO("With error estimate");
    impl(
        stepper.order(), stepper.number_of_substeps_for_error(),
        [&](const gsl::not_null<double*> u,
            const gsl::not_null<TimeSteppers::History<double>*> history,
            const TimeDelta& time_step) {
          double error;
          stepper.update_u(u, make_not_null(&error), history, time_step);
        },
        [&](const TimeStepId& time_step_id, const TimeDelta& time_step) {
          return stepper.next_time_id_for_error(time_step_id, time_step);
        });
  }
}
}  // namespace TimeStepperTestUtils
