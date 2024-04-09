// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

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
#include "Utilities/Literals.hpp"

namespace {
template <bool Monotonic>
void test_am() {
  for (size_t order = 2; order < 9; ++order) {
    CAPTURE(order);
    const TimeSteppers::AdamsMoultonPc<Monotonic> stepper(order);
    CHECK(stepper.order() == order);
    CHECK(stepper.error_estimate_order() == order - 1);
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
    for (size_t history_order = 2; history_order <= order; ++history_order) {
      CAPTURE(history_order);
      std::pair<int32_t, int32_t> convergence_step_range{10, 30};
      int32_t stride = 1;
      if (Monotonic) {
        // Monotonic dense output is much noisier.
        convergence_step_range.second =
            110 - 10 * static_cast<int32_t>(history_order);
        stride = 3 - static_cast<int32_t>(history_order) / 3;
      }
      TimeStepperTestUtils::check_dense_output(stepper, history_order,
                                               convergence_step_range, stride,
                                               not Monotonic);
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
    const auto time_step = slab.duration() / 2;
    const TimeStepId step_id(true, 1, slab.start() + time_step);
    const TimeSteppers::AdamsMoultonPc<Monotonic> stepper(2);
    TimeSteppers::History<double> history(2);
    history.insert(TimeStepId(true, 1, slab.start()), 0., 0.);
    history.insert(step_id, 0., 0.);
    CHECK_FALSE(stepper.can_change_step_size(
        step_id.next_substep(time_step, 1.0), history));
  }

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

// [[Timeout, 10]]
SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsMoultonPc", "[Unit][Time]") {
  test_am<false>();
}

// [[Timeout, 10]]
SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsMoultonPcMonotonic",
                  "[Unit][Time]") {
  test_am<true>();
}
}  // namespace
