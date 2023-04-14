// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Time/ApproximateTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Utilities/Rational.hpp"

namespace {
namespace ac = TimeSteppers::adams_coefficients;

void check_consistency(const ac::OrderVector<Time>& control_times,
                       const ac::OrderVector<double>& standard_coefficients) {
  ac::OrderVector<double> control_times_for_variable{};
  for (const Time& t : control_times) {
    control_times_for_variable.push_back(t.value());
  }

  CHECK_ITERABLE_APPROX(
      ac::variable_coefficients(control_times_for_variable,
                                control_times_for_variable.back(), 0.0),
      standard_coefficients);
  // This should be exact, because the function should detect the
  // constant step case and forward to that function.  (And then
  // multiply by 1, but that's also exact.)
  CHECK(ac::coefficients(control_times.begin(), control_times.end(),
                         ApproximateTimeDelta{1.0}) == standard_coefficients);

  // Test offset times
  {
    const double offset = 3.3;
    ac::OrderVector<Time> offset_control_times{};
    ac::OrderVector<double> offset_control_times_for_variable{};
    for (const Time& t : control_times) {
      offset_control_times.emplace_back(Slab(t.slab().start().value() + offset,
                                             t.slab().end().value() + offset),
                                        t.fraction());
      offset_control_times_for_variable.push_back(
          offset_control_times.back().value());
    }
    CHECK(ac::coefficients(offset_control_times.begin(),
                           offset_control_times.end(),
                           ApproximateTimeDelta{1.0}) == standard_coefficients);
    CHECK_ITERABLE_APPROX(ac::variable_coefficients(
                              offset_control_times_for_variable,
                              offset_control_times_for_variable.back(), offset),
                          standard_coefficients);
  }

  // Test scaling with time step
  {
    const double time_step = 2.1;
    ac::OrderVector<Time> scaled_control_times{};
    ac::OrderVector<double> scaled_control_times_for_variable{};
    for (const Time& t : control_times) {
      scaled_control_times.emplace_back(
          Slab(t.slab().start().value() * time_step,
               t.slab().end().value() * time_step),
          t.fraction());
      scaled_control_times_for_variable.push_back(
          scaled_control_times.back().value());
    }
    const auto scaled_coefficients = ac::coefficients(
        scaled_control_times.begin(), scaled_control_times.end(),
        ApproximateTimeDelta{time_step});
    const auto scaled_variable_coefficients = ac::variable_coefficients(
        scaled_control_times_for_variable,
        scaled_control_times_for_variable.back(), 0.0);
    for (size_t i = 0; i < standard_coefficients.size(); ++i) {
      CHECK(scaled_coefficients[i] == time_step * standard_coefficients[i]);
      CHECK(scaled_variable_coefficients[i] ==
            approx(time_step * standard_coefficients[i]));
    }
  }
}

void check_adams_bashforth_consistency() {
  ac::OrderVector<Time> standard_ab_control_times{};
  for (size_t order = 1; order <= ac::maximum_order; ++order) {
    standard_ab_control_times.insert(
        standard_ab_control_times.begin(),
        Slab::with_duration_from_start(-static_cast<double>(order), 1.0)
            .start());
    check_consistency(standard_ab_control_times,
                      ac::constant_adams_bashforth_coefficients(order));
  }
}

void test_rational_computation() {
  // Check a few known cases

  // Euler's method
  CHECK(ac::variable_coefficients<Rational>({0}, 0, 1) ==
        ac::OrderVector<Rational>{1});
  CHECK(ac::variable_coefficients<Rational>({0}, 0, 2) ==
        ac::OrderVector<Rational>{2});
  // Backwards Euler method
  CHECK(ac::variable_coefficients<Rational>({0}, -1, 0) ==
        ac::OrderVector<Rational>{1});
  CHECK(ac::variable_coefficients<Rational>({0}, -2, 0) ==
        ac::OrderVector<Rational>{2});
  // AB3
  CHECK(ac::variable_coefficients<Rational>({-3, -2, -1}, -1, 0) ==
        ac::OrderVector<Rational>{{5, 12}, {-4, 3}, {23, 12}});
  // AB3 backwards
  CHECK(ac::variable_coefficients<Rational>({3, 2, 1}, 1, 0) ==
        ac::OrderVector<Rational>{{-5, 12}, {4, 3}, {-23, 12}});
  // AM3
  CHECK(ac::variable_coefficients<Rational>({-2, -1, 0}, -1, 0) ==
        ac::OrderVector<Rational>{{-1, 12}, {2, 3}, {5, 12}});
  // AM3 backwards
  CHECK(ac::variable_coefficients<Rational>({2, 1, 0}, 1, 0) ==
        ac::OrderVector<Rational>{{1, 12}, {-2, 3}, {-5, 12}});
  // Variable step case
  CHECK(ac::variable_coefficients<Rational>({{-5, 2}, {-2, 3}}, {-2, 3}, 0) ==
        ac::OrderVector<Rational>{{-4, 33}, {26, 33}});
  // Step not aligned with control points
  CHECK(ac::variable_coefficients<Rational>({{1, 2}, {3, 4}}, {1, 3}, 1) ==
        ac::OrderVector<Rational>{{2, 9}, {4, 9}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsCoefficients", "[Unit][Time]") {
  // These tests just do consistency checks in the various functions.
  // The actual values are tested by the time stepper tests.
  check_adams_bashforth_consistency();

  test_rational_computation();
}
