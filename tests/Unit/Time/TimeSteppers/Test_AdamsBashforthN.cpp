// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <cmath>
#include <deque>

#include "DataStructures/MakeWithValue.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/Time/TimeSteppers/TimeStepperTestUtils.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN", "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order, false);
    TimeStepperTestUtils::check_multistep_properties(stepper);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::integrate_test(stepper, 1., epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const TimeSteppers::AdamsBashforthN stepper(order, true);
    TimeStepperTestUtils::check_multistep_properties(stepper);
    // Accuracy limited by first step
    TimeStepperTestUtils::integrate_test(stepper, 1., 1e-3);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Variable",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::integrate_variable_test(
        TimeSteppers::AdamsBashforthN(order, false), epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    // Accuracy limited by first step
    TimeStepperTestUtils::integrate_variable_test(
        TimeSteppers::AdamsBashforthN(order, true), 1e-3);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Backwards",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::integrate_test(
        TimeSteppers::AdamsBashforthN(order, false), -1., epsilon);
  }

  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    // Accuracy limited by first step
    TimeStepperTestUtils::integrate_test(
        TimeSteppers::AdamsBashforthN(order, true), -1., 1e-3);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Stability",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    TimeStepperTestUtils::stability_test(
        TimeSteppers::AdamsBashforthN(order, false));
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Factory",
                  "[Unit][Time]") {
  test_factory_creation<TimeStepper>("  AdamsBashforthN:\n"
                                     "    TargetOrder: 3");
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.Equal",
                  "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::equal_rate_boundary(
        TimeSteppers::AdamsBashforthN(order, false), epsilon, true);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.Equal.Backwards",
    "[Unit][Time]") {
  for (size_t order = 1; order < 9; ++order) {
    INFO(order);
    const double epsilon = std::max(std::pow(1e-3, order), 1e-14);
    TimeStepperTestUtils::equal_rate_boundary(
        TimeSteppers::AdamsBashforthN(order, false), epsilon, false);
  }
}

namespace {
// Non-copyable double to verify that the boundary code is not making
// internal copies.
class NCd {
 public:
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

NCd operator*(double a, NCd&& b) { return NCd(a * b()); }
NCd& operator+=(NCd& a, NCd&& b) { return a = NCd(a() + b()); }

// Random numbers
constexpr double c10 = 0.949716728952811;
constexpr double c11 = 0.190663110072823;
constexpr double c20 = 0.932407227651314;
constexpr double c21 = 0.805454101952822;
constexpr double c22 = 0.825876851406978;

// Test coupling for integrating using two drivers (multiplied together)
NCd quartic_coupling(const std::vector<std::reference_wrapper<const NCd>>& v) {
  CHECK(v.size() == 3);
  return NCd(v[1]() * v[2]());
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

// Analytic solutions
const std::array<double (*const)(double), 3> side_solutions{{
    quartic_answer, quartic_side1, quartic_side2}};

// Value to assign to points during a simulation
double simulation_value(const size_t side, const Time& t, const double y) {
  switch (side) {
    case 0: return y;
    case 1: return quartic_side1(t.value());
    case 2: return quartic_side2(t.value());
    default: ERROR("");
  };
}
}  // namespace

namespace MakeWithValueImpls {
template <>
SPECTRE_ALWAYS_INLINE NCd MakeWithValueImpl<NCd, NCd>::apply(
    const NCd& /*unused*/, double value) {
  return NCd(value);
}
}  // namespace MakeWithValueImpls

SPECTRE_TEST_CASE(
    "Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.MultipleSides",
    "[Unit][Time]") {
  constexpr double unused = 4444.;

  const Slab slab(0., 1.);

  Time t = slab.start();
  const TimeDelta dt = slab.duration() / 4;

  TimeSteppers::AdamsBashforthN ab4(4, false);

  std::vector<std::deque<std::tuple<Time, NCd, NCd>>> history(3);

  {
    const Slab init_slab = slab.retreat();
    const TimeDelta init_dt = dt.with_slab(init_slab);

    for (ssize_t step = 1; step <= 3; ++step) {
      const Time now = t - step * init_dt;
      for (size_t side = 0; side < 3; ++side) {
        history[side].emplace_front(
            now, NCd(unused), NCd(gsl::at(side_solutions, side)(now.value())));
      }
    }
  }

  double y = 0.;
  while (t < slab.end()) {
    for (size_t side = 0; side < 3; ++side) {
      history[side].emplace_back(t, NCd(unused),
                                 NCd(simulation_value(side, t, y)));
    }

    t += dt;
    y += ab4.compute_boundary_delta(
        quartic_coupling, make_not_null(&history), dt)();
    CHECK(y == approx(quartic_answer(t.value())));
  }
}

namespace {
void do_lts_test(const std::array<TimeDelta, 3>& dt) noexcept {
  constexpr double unused = 4444.;

  const bool forward_in_time = dt[0].is_positive();
  const auto simulation_less =
      [forward_in_time](const Time& a, const Time& b) noexcept {
    return forward_in_time ? a < b : b < a;
  };

  const Slab slab = dt[0].slab();
  Time t = forward_in_time ? slab.start() : slab.end();

  TimeSteppers::AdamsBashforthN ab4(4, false);

  std::vector<std::deque<std::tuple<Time, NCd, NCd>>> history(3);
  {
    const Slab init_slab = slab.advance_towards(-dt[0]);

    for (ssize_t step = 1; step <= 3; ++step) {
      for (size_t side = 0; side < 3; ++side) {
        const Time now = t - step * gsl::at(dt, side).with_slab(init_slab);
        history[side].emplace_front(
            now, NCd(unused), NCd(gsl::at(side_solutions, side)(now.value())));
      }
    }
  }

  double y = quartic_answer(t.value());
  Time next_check = t + dt[0];
  std::array<Time, 3> next{{t, t, t}};
  for (;;) {
    const auto side = static_cast<size_t>(
        std::min_element(next.cbegin(), next.cend(), simulation_less)
        - next.cbegin());

    history[side].emplace_back(t, NCd(unused),
                               NCd(simulation_value(side, t, y)));

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
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.LocalStepping",
    "[Unit][Time]") {
  const Slab slab(0., 1.);
  const TimeDelta full = slab.duration();
  do_lts_test({{full / 4, full / 4, full / 4}});
  do_lts_test({{full / 4, full / 8, full / 8}});
  do_lts_test({{full / 4, full / 8, full / 4}});
  do_lts_test({{full / 8, full / 4, full / 4}});
  do_lts_test({{full / 8, full / 4, full / 8}});
  do_lts_test({{full / 4, full / 8, full / 16}});
  do_lts_test({{full / 16, full / 4, full / 8}});
  do_lts_test({{full / 8, full / 16, full / 4}});

  // Non-nesting cases
  do_lts_test({{full / 4, full / 6, full / 8}});
  do_lts_test({{full / 6, full / 8, full / 4}});
  do_lts_test({{full / 8, full / 4, full / 6}});
  do_lts_test({{full / 5, full / 7, full / 13}});
  do_lts_test({{full / 7, full / 13, full / 5}});
  do_lts_test({{full / 13, full / 5, full / 7}});
}

SPECTRE_TEST_CASE(
    "Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.LocalSteppingBackward",
    "[Unit][Time]") {
  const Slab slab(0., 1.);
  const TimeDelta full = -slab.duration();
  do_lts_test({{full / 4, full / 4, full / 4}});
  do_lts_test({{full / 4, full / 8, full / 8}});
  do_lts_test({{full / 4, full / 8, full / 4}});
  do_lts_test({{full / 8, full / 4, full / 4}});
  do_lts_test({{full / 8, full / 4, full / 8}});
  do_lts_test({{full / 4, full / 8, full / 16}});
  do_lts_test({{full / 16, full / 4, full / 8}});
  do_lts_test({{full / 8, full / 16, full / 4}});

  // Non-nesting cases
  do_lts_test({{full / 4, full / 6, full / 8}});
  do_lts_test({{full / 6, full / 8, full / 4}});
  do_lts_test({{full / 8, full / 4, full / 6}});
  do_lts_test({{full / 5, full / 7, full / 13}});
  do_lts_test({{full / 7, full / 13, full / 5}});
  do_lts_test({{full / 13, full / 5, full / 7}});
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Boundary.Variable",
                  "[Unit][Time]") {
  constexpr double unused = 4444.;

  const Slab slab(0., 1.);

  Time t = slab.start();

  TimeSteppers::AdamsBashforthN ab4(4, false);

  std::vector<std::deque<std::tuple<Time, NCd, NCd>>> history(3);
  {
    const Slab init_slab = slab.retreat();
    const TimeDelta init_dt = init_slab.duration() / 4;

    // clang-tidy misfeature: warns about boost internals here
    for (ssize_t step = 1; step <= 3; ++step) {  // NOLINT
      // clang-tidy misfeature: warns about boost internals here
      const Time now = t - step * init_dt;  // NOLINT
      for (size_t side = 0; side < 3; ++side) {
        history[side].emplace_front(
            now, NCd(unused), NCd(gsl::at(side_solutions, side)(now.value())));
      }
    }
  }

  std::array<std::deque<TimeDelta>, 3> dt{{
      {slab.duration() / 4, slab.duration() / 2, slab.duration() / 4},
      {slab.duration() / 3, slab.duration() / 3, slab.duration() / 3},
      {slab.duration() / 6, slab.duration() / 6, slab.duration() * 2 / 9,
            slab.duration() * 4 / 9}}};

  double y = quartic_answer(t.value());
  Time next_check = t + dt[0][0];
  std::array<Time, 3> next{{t, t, t}};
  for (;;) {
    const auto side = static_cast<size_t>(
        std::min_element(next.cbegin(), next.cend()) - next.cbegin());

    history[side].emplace_back(gsl::at(next, side), NCd(unused),
                               NCd(simulation_value(side, gsl::at(next, side),
                                                    y)));

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

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsBashforthN.Serialization",
                  "[Unit][Time]") {
  TimeSteppers::AdamsBashforthN ab(4, false);
  test_serialization(ab);
  test_serialization_via_base<TimeStepper, TimeSteppers::AdamsBashforthN>(
      4_st, false);
  // test operator !=
  TimeSteppers::AdamsBashforthN ab2(4, true);
  CHECK(ab != ab2);
}
