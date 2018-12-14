// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>

#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order);
    TimeStepperTestUtils::check_multistep_properties(stepper);
    for (size_t start_points = 0; start_points < order; ++start_points) {
      INFO(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::integrate_test(stepper, start_points, 1., epsilon);
    }
    TimeStepperTestUtils::check_convergence_order(stepper, order);
    TimeStepperTestUtils::check_dense_output(stepper, order);
  }

  const Slab slab(0., 1.);
  const Time start = slab.start();
  const Time mid = start + slab.duration() / 2;
  const Time end = slab.end();
  const auto can_change = [](const Time& first, const Time& second,
                             const Time& now) noexcept {
    const TimeSteppers::AdamsBashforthN stepper(2);
    TimeSteppers::History<double, double> history;
    history.insert(first, 0., 0.);
    history.insert(second, 0., 0.);
    return stepper.can_change_step_size(TimeId(true, 0, now), history);
  };
  CHECK(can_change(start, mid, end));
  CHECK_FALSE(can_change(start, end, mid));
  CHECK_FALSE(can_change(mid, start, end));
  CHECK_FALSE(can_change(mid, end, start));
  CHECK_FALSE(can_change(end, start, mid));
  CHECK_FALSE(can_change(end, mid, start));

  test_factory_creation<TimeStepper>("  AdamsBashforthN:\n"
                                     "    Order: 3");
  test_factory_creation<LtsTimeStepper>("  AdamsBashforthN:\n"
                                        "    Order: 3");

  TimeSteppers::AdamsBashforthN ab4(4);
  test_serialization(ab4);
  test_serialization_via_base<TimeStepper, TimeSteppers::AdamsBashforthN>(4_st);
  test_serialization_via_base<LtsTimeStepper, TimeSteppers::AdamsBashforthN>(
      4_st);
  // test operator !=
  TimeSteppers::AdamsBashforthN ab2(2);
  CHECK(ab4 != ab2);
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Variable",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    for (size_t start_points = 0; start_points < order; ++start_points) {
      INFO(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::integrate_variable_test(
          TimeSteppers::AdamsBashforthN(order), start_points, epsilon);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Backwards",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    for (size_t start_points = 0; start_points < order; ++start_points) {
      INFO(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::integrate_test(
          TimeSteppers::AdamsBashforthN(order), start_points, -1., epsilon);
    }
  }

  const Slab slab(0., 1.);
  const Time start = slab.start();
  const Time mid = start + slab.duration() / 2;
  const Time end = slab.end();
  const auto can_change = [](const Time& first, const Time& second,
                             const Time& now) noexcept {
    const TimeSteppers::AdamsBashforthN stepper(2);
    TimeSteppers::History<double, double> history;
    history.insert(first, 0., 0.);
    history.insert(second, 0., 0.);
    return stepper.can_change_step_size(TimeId(false, 0, now), history);
  };
  CHECK_FALSE(can_change(start, mid, end));
  CHECK_FALSE(can_change(start, end, mid));
  CHECK_FALSE(can_change(mid, start, end));
  CHECK_FALSE(can_change(mid, end, start));
  CHECK_FALSE(can_change(end, start, mid));
  CHECK(can_change(end, mid, start));
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Stability",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    TimeStepperTestUtils::stability_test(TimeSteppers::AdamsBashforthN(order));
  }
}

namespace {
// Non-copyable double to verify that the boundary code is not making
// internal copies.
class NCd {
 public:
  NCd() = default;
  explicit NCd(double x) : x_(x) {}
  NCd(const NCd&) = delete;
  NCd(NCd&&) = default;
  NCd& operator=(const NCd&) = delete;
  NCd& operator=(NCd&&) = default;
  ~NCd() = default;

  double operator()() const { return x_; }

 private:
  double x_;
};

NCd operator*(double a, const NCd& b) { return NCd(a * b()); }
NCd& operator+=(NCd& a, NCd&& b) { return a = NCd(a() + b()); }
NCd& operator*=(NCd& a, double b) { return a = NCd(a() * b); }

// Random numbers
constexpr double c10 = 0.949716728952811;
constexpr double c11 = 0.190663110072823;
constexpr double c20 = 0.932407227651314;
constexpr double c21 = 0.805454101952822;
constexpr double c22 = 0.825876851406978;

// Test coupling for integrating using two drivers (multiplied together)
NCd quartic_coupling(const NCd& local, const NCd& remote) {
  return NCd(local() * remote());
}

// Test functions for integrating a quartic using the above coupling.
// The derivative of quartic_answer is the product of the other two.
double quartic_side1(double x) { return c10 + x * c11; }
double quartic_side2(double x) { return c20 + x * (c21 + x * c22); }
double quartic_answer(double x) {
  return x * (c10 * c20
              + x * ((c10 * c21 + c11 * c20) / 2
                     + x * ((c10 * c22 + c11 * c21) / 3
                            + x * (c11 * c22 / 4))));
}
}  // namespace

namespace MakeWithValueImpls {
template <>
template <typename ValueType>
SPECTRE_ALWAYS_INLINE NCd MakeWithValueImpl<NCd, NCd>::apply(
    const NCd& /*unused*/, ValueType value) noexcept {
  return NCd(value);
}
}  // namespace MakeWithValueImpls

namespace {
void do_lts_test(const std::array<TimeDelta, 2>& dt) noexcept {
  // For general time steppers the boundary stepper cannot be run
  // without simultaneously running the volume stepper.  For
  // Adams-Bashforth methods, however, the volume contribution is zero
  // if all the derivative contributions are from the boundary, so we
  // can leave it out.

  const bool forward_in_time = dt[0].is_positive();
  const auto simulation_less =
      [forward_in_time](const Time& a, const Time& b) noexcept {
    return forward_in_time ? a < b : b < a;
  };

  const auto make_time_id = [forward_in_time](const Time& t) noexcept {
    return TimeId(forward_in_time, 0, t);
  };

  const Slab slab = dt[0].slab();
  Time t = forward_in_time ? slab.start() : slab.end();

  TimeSteppers::AdamsBashforthN ab4(4);

  TimeSteppers::BoundaryHistory<NCd, NCd, NCd> history;
  {
    const Slab init_slab = slab.advance_towards(-dt[0]);

    for (int32_t step = 1; step <= 3; ++step) {
      {
        const Time now = t - step * dt[0].with_slab(init_slab);
        history.local_insert_initial(make_time_id(now),
                                     NCd(quartic_side1(now.value())));
      }
      {
        const Time now = t - step * dt[1].with_slab(init_slab);
        history.remote_insert_initial(make_time_id(now),
                                      NCd(quartic_side2(now.value())));
      }
    }
  }

  double y = quartic_answer(t.value());
  Time next_check = t + dt[0];
  std::array<Time, 2> next{{t, t}};
  for (;;) {
    const auto side = static_cast<size_t>(
        std::min_element(next.cbegin(), next.cend(), simulation_less)
        - next.cbegin());

    if (side == 0) {
      history.local_insert(make_time_id(t), NCd(quartic_side1(t.value())));
    } else {
      history.remote_insert(make_time_id(t), NCd(quartic_side2(t.value())));
    }

    gsl::at(next, side) += gsl::at(dt, side);

    t = *std::min_element(next.cbegin(), next.cend(), simulation_less);

    ASSERT(not simulation_less(next_check, t), "Screwed up arithmetic");
    if (t == next_check) {
      y += ab4.compute_boundary_delta(
          quartic_coupling, make_not_null(&history), dt[0])();
      CHECK(y == approx(quartic_answer(t.value())));
      if (t.is_at_slab_boundary()) {
        break;
      }
      next_check += dt[0];
    }
  }
}

void check_lts_vts() noexcept {
  const Slab slab(0., 1.);

  const auto make_time_id = [](const Time& t) noexcept {
    return TimeId(true, 0, t);
  };

  Time t = slab.start();

  TimeSteppers::AdamsBashforthN ab4(4);

  TimeSteppers::BoundaryHistory<NCd, NCd, NCd> history;
  {
    const Slab init_slab = slab.retreat();
    const TimeDelta init_dt = init_slab.duration() / 4;

    // clang-tidy misfeature: warns about boost internals here
    for (int32_t step = 1; step <= 3; ++step) {  // NOLINT
      // clang-tidy misfeature: warns about boost internals here
      const Time now = t - step * init_dt;  // NOLINT
      history.local_insert_initial(make_time_id(now),
                                   NCd(quartic_side1(now.value())));
      history.remote_insert_initial(make_time_id(now),
                                    NCd(quartic_side2(now.value())));
    }
  }

  std::array<std::deque<TimeDelta>, 2> dt{{
      {slab.duration() / 2, slab.duration() / 4, slab.duration() / 4},
      {slab.duration() / 6, slab.duration() / 6, slab.duration() * 2 / 9,
            slab.duration() * 4 / 9}}};

  double y = quartic_answer(t.value());
  Time next_check = t + dt[0][0];
  std::array<Time, 2> next{{t, t}};
  for (;;) {
    const auto side = static_cast<size_t>(
        std::min_element(next.cbegin(), next.cend()) - next.cbegin());

    if (side == 0) {
      history.local_insert(make_time_id(next[0]),
                           NCd(quartic_side1(next[0].value())));
    } else {
      history.remote_insert(make_time_id(next[1]),
                            NCd(quartic_side2(next[1].value())));
    }

    const TimeDelta this_dt = gsl::at(dt, side).front();
    gsl::at(dt, side).pop_front();

    gsl::at(next, side) += this_dt;

    if (*std::min_element(next.cbegin(), next.cend()) == next_check) {
      y += ab4.compute_boundary_delta(
          quartic_coupling, make_not_null(&history), next_check - t)();
      CHECK(y == approx(quartic_answer(next_check.value())));
      if (next_check.is_at_slab_boundary()) {
        break;
      }
      t = next_check;
      next_check += dt[0].front();
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Boundary",
                  "[Unit][Time]") {
  // No local stepping
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order);
    for (size_t start_points = 0; start_points < order; ++start_points) {
      INFO(start_points);
      const double epsilon = std::max(std::pow(1e-3, start_points + 1), 1e-14);
      TimeStepperTestUtils::equal_rate_boundary(stepper, start_points, epsilon,
                                                true);
      TimeStepperTestUtils::equal_rate_boundary(stepper, start_points, epsilon,
                                                false);
    }
  }

  // Local stepping with constant step sizes
  const Slab slab(0., 1.);
  for (const auto full : {slab.duration(), -slab.duration()}) {
    do_lts_test({{full / 4, full / 4}});
    do_lts_test({{full / 4, full / 8}});
    do_lts_test({{full / 8, full / 4}});
    do_lts_test({{full / 16, full / 4}});
    do_lts_test({{full / 4, full / 16}});

    // Non-nesting cases
    do_lts_test({{full / 4, full / 6}});
    do_lts_test({{full / 6, full / 4}});
    do_lts_test({{full / 5, full / 7}});
    do_lts_test({{full / 7, full / 5}});
    do_lts_test({{full / 5, full / 13}});
    do_lts_test({{full / 13, full / 5}});
  }

  // Local stepping with varying time steps
  check_lts_vts();

  // Dense output
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    TimeStepperTestUtils::check_boundary_dense_output(
        TimeSteppers::AdamsBashforthN(order));
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Reversal",
                  "[Unit][Time]") {
  const TimeSteppers::AdamsBashforthN ab3(3);

  const auto f = [](const double t) noexcept {
    return 1. + t * (2. + t * (3. + t * 4.));
  };
  const auto df = [](const double t) noexcept {
    return 2. + t * (6. + t * 12.);
  };

  const Slab slab(0., 1.);
  TimeSteppers::History<double, double> history{};
  const auto add_history = [&df, &f, &history](const Time& time) noexcept {
    history.insert(time, f(time.value()), df(time.value()));
  };
  add_history(slab.start());
  add_history(slab.end());
  add_history(slab.start() + slab.duration() / 3);
  double y = f(1. / 3.);
  ab3.update_u(make_not_null(&y), make_not_null(&history), slab.duration() / 3);
  CHECK(y == approx(f(2. / 3.)));
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.Reversal",
                  "[Unit][Time]") {
  const TimeSteppers::AdamsBashforthN ab3(3);

  const auto f = [](const double t) noexcept {
    return 1. + t * (2. + t * (3. + t * 4.));
  };
  const auto df = [](const double t) noexcept {
    return 2. + t * (6. + t * 12.);
  };

  const Slab slab(0., 1.);
  TimeSteppers::BoundaryHistory<double, double, double> history{};
  const auto add_history = [&df, &history](const TimeId& time_id) noexcept {
    history.local_insert(time_id, df(time_id.time().value()));
    history.remote_insert(time_id, 0.);
  };
  add_history(TimeId(true, 0, slab.start()));
  add_history(TimeId(true, 0, slab.end()));
  add_history(TimeId(true, 1, slab.start() + slab.duration() / 3));
  double y = f(1. / 3.);
  y += ab3.compute_boundary_delta(
      [](const double local, const double /*remote*/) noexcept {
        return local;
      },
      make_not_null(&history), slab.duration() / 3);
  CHECK(y == approx(f(2. / 3.)));
}
