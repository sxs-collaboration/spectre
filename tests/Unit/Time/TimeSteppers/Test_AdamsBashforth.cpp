// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/LtsHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforth", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    CAPTURE(order);
    const TimeSteppers::AdamsBashforth stepper(order);
    TimeStepperTestUtils::check_multistep_properties(stepper);
    CHECK(stepper.monotonic());
    for (size_t start_points = 0; start_points < order; ++start_points) {
      CAPTURE(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::integrate_test(stepper, start_points + 1,
                                           start_points, 1., epsilon);
      TimeStepperTestUtils::integrate_test_explicit_time_dependence(
          stepper, start_points + 1, start_points, 1., epsilon);

      const double large_step_epsilon =
          std::clamp(1.0e3 * std::pow(2.0e-2, start_points + 1), 1e-14, 1.0);
      TimeStepperTestUtils::integrate_error_test(
          stepper, start_points + 1, start_points, 1.0, large_step_epsilon, 20,
          1.0e-4);
      TimeStepperTestUtils::integrate_error_test(
          stepper, start_points + 1, start_points, -1.0, large_step_epsilon, 20,
          1.0e-4);
    }
    TimeStepperTestUtils::check_convergence_order(stepper, {10, 30});
    TimeStepperTestUtils::check_dense_output(stepper, {10, 30}, 1, true);

    CHECK(stepper.order() == order);

    TimeStepperTestUtils::stability_test(stepper);
  }

  const Slab slab(0., 1.);
  const Time start = slab.start();
  const Time mid = slab.start() + slab.duration() / 2;
  const Time end = slab.end();
  const auto can_change = [](const Time& first, const Time& second,
                             const Time& now) {
    const TimeSteppers::AdamsBashforth stepper(2);
    TimeSteppers::History<double> history(2);
    history.insert(TimeStepId(true, 0, first), 0., 0.);
    history.insert(TimeStepId(true, 2, second), 0., 0.);
    return stepper.can_change_step_size(TimeStepId(true, 4, now), history);
  };
  CHECK(can_change(start, mid, end));
  CHECK_FALSE(can_change(start, end, mid));
  CHECK(can_change(mid, start, end));
  CHECK_FALSE(can_change(mid, end, start));
  CHECK_FALSE(can_change(end, start, mid));
  CHECK_FALSE(can_change(end, mid, start));

  TestHelpers::test_factory_creation<TimeStepper, TimeSteppers::AdamsBashforth>(
      "AdamsBashforth:\n"
      "  Order: 3");
  TestHelpers::test_factory_creation<LtsTimeStepper,
                                     TimeSteppers::AdamsBashforth>(
      "AdamsBashforth:\n"
      "  Order: 3");

  TimeSteppers::AdamsBashforth ab4(4);
  test_serialization(ab4);
  test_serialization_via_base<TimeStepper, TimeSteppers::AdamsBashforth>(4_st);
  test_serialization_via_base<LtsTimeStepper, TimeSteppers::AdamsBashforth>(
      4_st);
  // test operator !=
  TimeSteppers::AdamsBashforth ab2(2);
  CHECK(ab4 != ab2);

  TimeStepperTestUtils::check_strong_stability_preservation(
      TimeSteppers::AdamsBashforth(1), 1.0);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforth.Variable",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    for (size_t start_points = 0; start_points < order; ++start_points) {
      INFO(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::integrate_variable_test(
          TimeSteppers::AdamsBashforth(order), start_points + 1, start_points,
          epsilon);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforth.Backwards",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    for (size_t start_points = 0; start_points < order; ++start_points) {
      INFO(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::integrate_test(
          TimeSteppers::AdamsBashforth(order), start_points + 1, start_points,
          -1., epsilon);
      TimeStepperTestUtils::integrate_test_explicit_time_dependence(
          TimeSteppers::AdamsBashforth(order), start_points + 1, start_points,
          -1., epsilon);
    }
  }

  const Slab slab(0., 1.);
  const Time start = slab.start();
  const Time mid = slab.start() + slab.duration() / 2;
  const Time end = slab.end();
  const auto can_change = [](const Time& first, const Time& second,
                             const Time& now) {
    const TimeSteppers::AdamsBashforth stepper(2);
    TimeSteppers::History<double> history(2);
    history.insert(TimeStepId(false, 0, first), 0., 0.);
    history.insert(TimeStepId(false, 2, second), 0., 0.);
    return stepper.can_change_step_size(TimeStepId(false, 4, now), history);
  };
  CHECK_FALSE(can_change(start, mid, end));
  CHECK_FALSE(can_change(start, end, mid));
  CHECK_FALSE(can_change(mid, start, end));
  CHECK(can_change(mid, end, start));
  CHECK_FALSE(can_change(end, start, mid));
  CHECK(can_change(end, mid, start));
}

namespace {
void test_neighbor_data_required() {
  // Test is order-independent
  const TimeSteppers::AdamsBashforth stepper(4);
  const Slab slab(0.0, 1.0);
  CHECK(not stepper.neighbor_data_required(TimeStepId(true, 0, slab.start()),
                                           TimeStepId(true, 0, slab.start())));
  CHECK(not stepper.neighbor_data_required(TimeStepId(true, 0, slab.start()),
                                           TimeStepId(true, 0, slab.end())));
  CHECK(stepper.neighbor_data_required(TimeStepId(true, 0, slab.end()),
                                       TimeStepId(true, 0, slab.start())));

  CHECK(not stepper.neighbor_data_required(TimeStepId(false, 0, slab.end()),
                                           TimeStepId(false, 0, slab.end())));
  CHECK(not stepper.neighbor_data_required(TimeStepId(false, 0, slab.end()),
                                           TimeStepId(false, 0, slab.start())));
  CHECK(stepper.neighbor_data_required(TimeStepId(false, 0, slab.start()),
                                       TimeStepId(false, 0, slab.end())));
}
}  // namespace

// [[Timeout, 10]]
SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforth.Boundary",
                  "[Unit][Time]") {
  test_neighbor_data_required();

  for (size_t order = 1; order < 9; ++order) {
    CAPTURE(order);
    const TimeSteppers::AdamsBashforth stepper(order);
    TimeStepperTestUtils::lts::test_equal_rate(stepper);
    TimeStepperTestUtils::lts::test_uncoupled(stepper, 1e-12);
    TimeStepperTestUtils::lts::test_conservation(stepper);
    // Only test convergence for low-order methods, since it's hard to
    // find parameters where the high-order ones are in the convergent
    // limit but not roundoff-dominated.
    if (order < 5) {
      TimeStepperTestUtils::lts::test_convergence(stepper, {20, 100}, 20);
      TimeStepperTestUtils::lts::test_dense_convergence(stepper, {40, 200}, 40);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforth.Reversal",
                  "[Unit][Time]") {
  const TimeSteppers::AdamsBashforth ab3(3);

  const auto f = [](const double t) {
    return 1. + t * (2. + t * (3. + t * 4.));
  };
  const auto df = [](const double t) { return 2. + t * (6. + t * 12.); };

  TimeSteppers::History<double> history{3};
  const auto add_history = [&df, &f, &history](const int64_t slab,
                                               const Time& time) {
    history.insert(TimeStepId(true, slab, time), f(time.value()),
                   df(time.value()));
  };
  const Slab slab(0., 1.);
  add_history(0, slab.start());
  add_history(0, slab.start() + slab.duration() * 3 / 4);
  add_history(1, slab.start() + slab.duration() / 3);
  double y = f(1. / 3.);
  ab3.update_u(make_not_null(&y), history, slab.duration() / 3);
  CHECK(y == approx(f(2. / 3.)));
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforth.Boundary.Reversal",
                  "[Unit][Time]") {
  const size_t order = 3;
  const TimeSteppers::AdamsBashforth ab3(order);

  const auto f = [](const double t) {
    return 1. + t * (2. + t * (3. + t * 4.));
  };
  const auto df = [](const double t) { return 2. + t * (6. + t * 12.); };

  const Slab slab(0., 1.);
  TimeSteppers::BoundaryHistory<double, double, double> history{};
  const auto add_history = [&df, &history](const TimeStepId& time_id) {
    history.local().insert(time_id, order, df(time_id.step_time().value()));
    history.remote().insert(time_id, order, 0.);
  };
  add_history(TimeStepId(true, 0, slab.start()));
  add_history(TimeStepId(true, 0, slab.start() + slab.duration() * 3 / 4));
  add_history(TimeStepId(true, 1, slab.start() + slab.duration() / 3));
  double y = f(1. / 3.);
  ab3.add_boundary_delta(
      &y, history, slab.duration() / 3,
      [](const double local, const double /*remote*/) { return local; });
  CHECK(y == approx(f(2. / 3.)));
}
