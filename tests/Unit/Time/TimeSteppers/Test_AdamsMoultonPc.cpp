// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Time/TimeSteppers/LtsHelpers.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Rational.hpp"

namespace {
template <bool Monotonic>
void test_am() {
  for (size_t order = 2; order < 9; ++order) {
    CAPTURE(order);
    const TimeSteppers::AdamsMoultonPc<Monotonic> stepper(order);
    CHECK(stepper.order() == order);
    CHECK(stepper.number_of_past_steps() == order - 2);
    CHECK(stepper.number_of_substeps() == 2);
    CHECK(stepper.number_of_substeps_for_error() == 2);

    for (size_t start_points = 0;
         start_points <= stepper.number_of_past_steps();
         ++start_points) {
      CAPTURE(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 2), 1e-14);
      TimeStepperTestUtils::integrate_test(stepper, start_points + 2,
                                           start_points, 1., epsilon);
      TimeStepperTestUtils::integrate_test_explicit_time_dependence(
          stepper, start_points + 2, start_points, 1., epsilon);

      const double large_step_epsilon =
          std::clamp(1.0e2 * std::pow(2.0e-2, start_points + 2), 1e-14, 1.0);
      TimeStepperTestUtils::integrate_error_test(
          stepper, start_points + 2, start_points, 1.0, large_step_epsilon, 20,
          1.0e-4);
      TimeStepperTestUtils::integrate_error_test(
          stepper, start_points + 2, start_points, -1.0, large_step_epsilon, 20,
          1.0e-4);

      TimeStepperTestUtils::integrate_variable_test(stepper, start_points + 2,
                                                    start_points, epsilon);
    }
    TimeStepperTestUtils::check_convergence_order(stepper, {10, 30});
    {
      std::pair<int32_t, int32_t> convergence_step_range{10, 30};
      int32_t stride = 1;
      if (Monotonic) {
        // Monotonic dense output is much noisier.
        convergence_step_range.second = 110 - 10 * static_cast<int32_t>(order);
        stride = 3 - static_cast<int32_t>(order) / 3;
      }
      TimeStepperTestUtils::check_dense_output(stepper, convergence_step_range,
                                               stride, not Monotonic);
    }
  }

  const Slab slab(0., 1.);
  const Time start = slab.start();
  const Time mid = slab.start() + slab.duration() / 2;
  const Time end = slab.end();
  const auto can_change = [](const bool time_runs_forward, const Time& first,
                             const Time& second, const Time& now) {
    const TimeSteppers::AdamsMoultonPc<Monotonic> stepper(2);
    TimeSteppers::History<double> history(2);
    history.insert(TimeStepId(time_runs_forward, 0, first), 0., 0.);
    history.insert(TimeStepId(time_runs_forward, 2, second), 0., 0.);
    return stepper.can_change_step_size(TimeStepId(time_runs_forward, 4, now),
                                        history);
  };
  CHECK(can_change(true, start, mid, end));
  CHECK_FALSE(can_change(true, start, end, mid));
  CHECK(can_change(true, mid, start, end));
  CHECK_FALSE(can_change(true, mid, end, start));
  CHECK_FALSE(can_change(true, end, start, mid));
  CHECK_FALSE(can_change(true, end, mid, start));
  CHECK(can_change(true, start, mid, mid));
  CHECK_FALSE(can_change(true, start, mid, start));

  CHECK(can_change(false, end, mid, start));
  CHECK_FALSE(can_change(false, end, start, mid));
  CHECK(can_change(false, mid, end, start));
  CHECK_FALSE(can_change(false, mid, start, end));
  CHECK_FALSE(can_change(false, start, end, mid));
  CHECK_FALSE(can_change(false, start, mid, end));
  CHECK(can_change(false, end, mid, mid));
  CHECK_FALSE(can_change(false, end, mid, end));

  {
    TimeSteppers::AdamsMoultonPc<Monotonic> am4(4);
    TimeSteppers::AdamsMoultonPc<Monotonic> am2(2);
    CHECK(am4 == am4);
    CHECK_FALSE(am4 != am4);
    CHECK(am4 != am2);
    CHECK_FALSE(am4 == am2);

    test_serialization(am4);
    test_serialization_via_base<TimeStepper,
                                TimeSteppers::AdamsMoultonPc<Monotonic>>(4_st);
  }

  {
    const std::string name =
        Monotonic ? "AdamsMoultonPcMonotonic" : "AdamsMoultonPc";
    const auto created = TestHelpers::test_factory_creation<
        TimeStepper, TimeSteppers::AdamsMoultonPc<Monotonic>>(name +
                                                              ":\n"
                                                              "  Order: 3");
    CHECK(created->order() == 3);
  }

  {
    const auto check_order = [](const size_t order, const double phase) {
      CAPTURE(order);
      TimeStepperTestUtils::stability_test(
          TimeSteppers::AdamsMoultonPc<Monotonic>(order), phase);
    };

    check_order(2, M_PI);
    check_order(3, 2.504);
    check_order(4, 2.347);
    check_order(5, 2.339);
    check_order(6, 2.368);
    check_order(7, 2.369);
    check_order(8, 2.364);
  }
}

template <bool Monotonic>
void test_neighbor_data_required(const bool time_runs_forward) {
  // Test is order-independent
  const TimeSteppers::AdamsMoultonPc<Monotonic> stepper(4);
  const Slab slab(0.0, 1.0);
  const auto long_step = (time_runs_forward ? 1 : -1) * slab.duration() / 2;
  const auto short_step = long_step / 2;

  const TimeStepId start(time_runs_forward, 1,
                         time_runs_forward ? slab.start() : slab.end());
  const TimeStepId long_sub = start.next_substep(long_step, 1.0);
  const TimeStepId short_sub1 = start.next_substep(short_step, 1.0);
  const TimeStepId middle(time_runs_forward, 1, start.step_time() + short_step);
  const TimeStepId short_sub2 = middle.next_substep(short_step, 1.0);
  const TimeStepId end(time_runs_forward, 1, start.step_time() + long_step);

  const double middle_time = middle.step_time().value();

  if constexpr (Monotonic) {
    CHECK(not stepper.neighbor_data_required(start, start));

    CHECK(stepper.neighbor_data_required(long_sub, start));
    CHECK(not stepper.neighbor_data_required(long_sub, long_sub));
    CHECK(stepper.neighbor_data_required(long_sub, short_sub1));
    CHECK(stepper.neighbor_data_required(long_sub, middle));
    CHECK(not stepper.neighbor_data_required(long_sub, short_sub2));

    CHECK(stepper.neighbor_data_required(short_sub1, start));
    CHECK(not stepper.neighbor_data_required(short_sub1, long_sub));
    CHECK(not stepper.neighbor_data_required(short_sub1, short_sub1));
    CHECK(not stepper.neighbor_data_required(short_sub1, middle));

    CHECK(stepper.neighbor_data_required(middle, start));
    CHECK(not stepper.neighbor_data_required(middle, long_sub));
    CHECK(stepper.neighbor_data_required(middle, short_sub1));
    CHECK(not stepper.neighbor_data_required(middle, middle));

    CHECK(stepper.neighbor_data_required(middle_time, start));
    CHECK(not stepper.neighbor_data_required(middle_time, long_sub));
    CHECK(stepper.neighbor_data_required(middle_time, short_sub1));
    CHECK(stepper.neighbor_data_required(middle_time, middle));
    CHECK(not stepper.neighbor_data_required(middle_time, short_sub2));

    CHECK(stepper.neighbor_data_required(short_sub2, start));
    CHECK(not stepper.neighbor_data_required(short_sub2, long_sub));
    CHECK(stepper.neighbor_data_required(short_sub2, short_sub1));
    CHECK(stepper.neighbor_data_required(short_sub2, middle));
    CHECK(not stepper.neighbor_data_required(short_sub2, short_sub2));

    CHECK(stepper.neighbor_data_required(end, start));
    CHECK(stepper.neighbor_data_required(end, long_sub));
    CHECK(stepper.neighbor_data_required(end, short_sub1));
    CHECK(stepper.neighbor_data_required(end, middle));
    CHECK(stepper.neighbor_data_required(end, short_sub2));
    CHECK(not stepper.neighbor_data_required(end, end));
  } else {
    CHECK(not stepper.neighbor_data_required(start, start));

    CHECK(stepper.neighbor_data_required(long_sub, start));
    CHECK(not stepper.neighbor_data_required(long_sub, long_sub));
    CHECK(not stepper.neighbor_data_required(long_sub, short_sub1));
    CHECK(not stepper.neighbor_data_required(long_sub, middle));
    CHECK(not stepper.neighbor_data_required(long_sub, short_sub2));

    CHECK(stepper.neighbor_data_required(short_sub1, start));
    CHECK(not stepper.neighbor_data_required(short_sub1, long_sub));
    CHECK(not stepper.neighbor_data_required(short_sub1, short_sub1));
    CHECK(not stepper.neighbor_data_required(short_sub1, middle));

    CHECK(stepper.neighbor_data_required(middle, start));
    CHECK(stepper.neighbor_data_required(middle, long_sub));
    CHECK(stepper.neighbor_data_required(middle, short_sub1));
    CHECK(not stepper.neighbor_data_required(middle, middle));

    CHECK(stepper.neighbor_data_required(middle_time, start));
    CHECK(stepper.neighbor_data_required(middle_time, long_sub));
    CHECK(stepper.neighbor_data_required(middle_time, short_sub1));
    CHECK(not stepper.neighbor_data_required(middle_time, middle));

    CHECK(stepper.neighbor_data_required(short_sub2, start));
    CHECK(stepper.neighbor_data_required(short_sub2, long_sub));
    CHECK(stepper.neighbor_data_required(short_sub2, short_sub1));
    CHECK(stepper.neighbor_data_required(short_sub2, middle));
    CHECK(not stepper.neighbor_data_required(short_sub2, short_sub2));

    CHECK(stepper.neighbor_data_required(end, start));
    CHECK(stepper.neighbor_data_required(end, long_sub));
    CHECK(stepper.neighbor_data_required(end, short_sub1));
    CHECK(stepper.neighbor_data_required(end, middle));
    CHECK(stepper.neighbor_data_required(end, short_sub2));
    CHECK(not stepper.neighbor_data_required(end, end));
  }
}

template <bool Monotonic>
void test_boundary() {
  test_neighbor_data_required<Monotonic>(true);
  test_neighbor_data_required<Monotonic>(false);

  for (size_t order = 2; order < 9; ++order) {
    CAPTURE(order);
    const TimeSteppers::AdamsMoultonPc<Monotonic> stepper(order);
    TimeStepperTestUtils::lts::test_equal_rate(stepper);
    TimeStepperTestUtils::lts::test_uncoupled(stepper, 1e-12);
    TimeStepperTestUtils::lts::test_conservation(stepper);
    // Only test convergence for low-order methods, since it's hard to
    // find parameters where the high-order ones are in the convergent
    // limit but not roundoff-dominated.
    if (order < 5) {
      TimeStepperTestUtils::lts::test_convergence(stepper, {20, 100}, 20);
      TimeStepperTestUtils::lts::test_dense_convergence(stepper, {50, 170}, 15);
    }
  }
}

TimeStepId make_id(const Rational& time_fraction, const uint64_t substep,
                   const Rational& step_fraction = 0) {
  ASSERT(substep == 0 or step_fraction != 0, "Need step size for a substep.");
  const Slab slab(0.0, 1.0);
  const TimeStepId step_id(true, 0, Time(slab, time_fraction));
  if (substep == 0) {
    return step_id;
  } else {
    return step_id.next_substep(TimeDelta(slab, step_fraction), 1.0);
  }
}

using Counts = std::vector<std::pair<Rational, size_t>>;

Counts substep_counts(const TimeSteppers::ConstBoundaryHistoryTimes& times) {
  std::vector<std::pair<Rational, size_t>> counts{};
  counts.reserve(times.size());
  for (const auto& step : times) {
    counts.emplace_back(step.step_time().fraction(),
                        times.number_of_substeps(step));
  }
  return counts;
}

void test_non_monotonic_boundary_cleaning() {
  const TimeSteppers::AdamsMoultonPc<false> am3(3);
  TimeSteppers::BoundaryHistory<double, double, double> history{};

  // Equal rate
  history.local().insert(make_id({0, 8}, 0), 3, 0.0);
  history.local().insert(make_id({1, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({0, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({1, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{0, 8}, 1}, {{1, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{0, 8}, 1}, {{1, 8}, 1}});
  history.local().insert(make_id({1, 8}, 1, {1, 8}), 3, 0.0);
  history.remote().insert(make_id({1, 8}, 1, {1, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{1, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}});

  // Local finer
  history.local().insert(make_id({2, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({2, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{1, 8}, 1}, {{2, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}, {{2, 8}, 1}});
  history.local().insert(make_id({2, 8}, 1, {1, 8}), 3, 0.0);
  history.remote().insert(make_id({2, 8}, 1, {2, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{2, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}, {{2, 8}, 2}});
  history.local().insert(make_id({3, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{2, 8}, 1}, {{3, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}, {{2, 8}, 2}});
  history.local().insert(make_id({3, 8}, 1, {1, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{3, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{2, 8}, 1}});

  // Local coarser
  history.local().insert(make_id({4, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({4, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{3, 8}, 1}, {{4, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{2, 8}, 1}, {{4, 8}, 1}});
  history.local().insert(make_id({4, 8}, 1, {2, 8}), 3, 0.0);
  history.remote().insert(make_id({4, 8}, 1, {1, 8}), 3, 0.0);
  history.remote().insert(make_id({5, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({5, 8}, 1, {1, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{4, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{5, 8}, 1}});
}

void test_monotonic_boundary_cleaning() {
  const TimeSteppers::AdamsMoultonPc<true> am3(3);
  TimeSteppers::BoundaryHistory<double, double, double> history{};

  // Equal rate
  history.local().insert(make_id({0, 8}, 0), 3, 0.0);
  history.local().insert(make_id({1, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({0, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({1, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{0, 8}, 1}, {{1, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{0, 8}, 1}, {{1, 8}, 1}});
  history.local().insert(make_id({1, 8}, 1, {1, 8}), 3, 0.0);
  history.remote().insert(make_id({1, 8}, 1, {1, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{1, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}});

  // Local finer
  history.local().insert(make_id({2, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({2, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{1, 8}, 1}, {{2, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}, {{2, 8}, 1}});
  history.local().insert(make_id({2, 8}, 1, {1, 8}), 3, 0.0);
  history.remote().insert(make_id({2, 8}, 1, {2, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{1, 8}, 1}, {{2, 8}, 2}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}, {{2, 8}, 2}});
  history.local().insert(make_id({3, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) ==
        Counts{{{1, 8}, 1}, {{2, 8}, 2}, {{3, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{1, 8}, 1}, {{2, 8}, 2}});
  history.local().insert(make_id({3, 8}, 1, {1, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{3, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{2, 8}, 1}});

  // Local coarser
  history.local().insert(make_id({4, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({4, 8}, 0), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{3, 8}, 1}, {{4, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{2, 8}, 1}, {{4, 8}, 1}});
  history.local().insert(make_id({4, 8}, 1, {2, 8}), 3, 0.0);
  history.remote().insert(make_id({4, 8}, 1, {1, 8}), 3, 0.0);
  history.remote().insert(make_id({5, 8}, 0), 3, 0.0);
  history.remote().insert(make_id({5, 8}, 1, {1, 8}), 3, 0.0);
  am3.clean_boundary_history(make_not_null(&history));
  CHECK(substep_counts(history.local()) == Counts{{{4, 8}, 1}});
  CHECK(substep_counts(history.remote()) == Counts{{{5, 8}, 1}});
}

// [[Timeout, 30]]
SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsMoultonPc", "[Unit][Time]") {
  test_am<false>();
  test_boundary<false>();
  test_non_monotonic_boundary_cleaning();
}

// [[Timeout, 30]]
SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsMoultonPcMonotonic",
                  "[Unit][Time]") {
  test_am<true>();
  test_boundary<true>();
  test_monotonic_boundary_cleaning();
}
}  // namespace
