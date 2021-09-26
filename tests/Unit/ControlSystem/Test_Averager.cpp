// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "ControlSystem/Averager.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t DerivOrder>
void test_linear() {
  INFO("Test linear.");
  double t = 0.0;
  const double dt = 0.1;
  const double final_time = 5.0;

  // test true and false `using_average_0th_deriv_of_q`
  Averager<DerivOrder> averager_t(0.5, true);
  Averager<DerivOrder> averager_f(0.5, false);

  CHECK(averager_t.using_average_0th_deriv_of_q());
  CHECK_FALSE(averager_f.using_average_0th_deriv_of_q());

  // define custom approx for higher derivative checks
  Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);

  while (t < final_time) {
    // test using an analytic function f(t) = t
    DataVector analytic_func{DerivOrder + 1, 0.0};
    analytic_func[0] = t;
    analytic_func[1] = 1.0;

    // update exponential averager
    averager_t.update(t, {analytic_func[0]}, {0.1});
    averager_f.update(t, {analytic_func[0]}, {0.1});
    // compare values once averager has sufficient data
    if (averager_t(t)) {
      const auto result_t = averager_t(t).value();
      // check function value, which should agree with the effective time
      CHECK(approx(result_t[0][0]) == averager_t.average_time(t));
      // check derivatives
      // The exponential averager uses finite differencing to approximate the
      // derivatives, which is why we enforce a less stringent check here.
      for (size_t i = 1; i < DerivOrder + 1; i++) {
        CHECK(custom_approx(gsl::at(result_t, i)[0]) == analytic_func[i]);
      }
    }
    if (averager_f(t)) {
      const auto result_f = averager_f(t).value();
      // check function value, which should agree with the true time `t`
      CHECK(approx(result_f[0][0]) == t);
      // check derivatives
      // The exponential averager uses finite differencing to approximate the
      // derivatives, which is why we enforce a less stringent check here.
      for (size_t i = 1; i < DerivOrder + 1; i++) {
        CHECK(custom_approx(gsl::at(result_f, i)[0]) == analytic_func[i]);
      }
    }

    t += dt;
  }
}

template <size_t DerivOrder>
void test_semianalytic() {
  INFO("Test semianalytic.");
  double t = 0.0;
  const double final_time = 5.0;

  // The equations suggests that our exponential averager is equivalent
  // to the simple exponential averaging method of
  // F_avg(t) = \alpha*F(t) + (1 - \alpha)*F_avg(t_{-1}}
  // where \alpha = \tau_m / (W(t)*D) and D = 1.0 + \tau_m/tau_avg
  // However, the exponential averager that we are testing has time varying
  // weight, so we allow \alpha to be a function of time and update at
  // each timestep.

  // some vars for a simple exponential average to compare against
  std::array<DataVector, DerivOrder + 1> avg_values =
      make_array<DerivOrder + 1, DataVector>(DataVector{1, 0.0});
  double avg_time = 0.0;

  // the measurement timescale (tau_m)
  const double tau_m = 0.1;
  const double avg_tscale_fac = 1.0;
  const double damping_time = 0.1;
  const double denom = 1.0 + tau_m / (avg_tscale_fac * damping_time);
  // the average weight, represents W(t) in the above comment
  double avg_weight = 0.0;

  Averager<DerivOrder> averager(avg_tscale_fac, true);

  // define custom approx for second derivative checks
  Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);

  while (t < final_time) {
    // test using an analytic function f(t) = t**DerivOrder
    DataVector analytic_func{DerivOrder + 1, 0.0};
    if constexpr (DerivOrder == 1) {
      analytic_func[0] = t;
      analytic_func[1] = 1.0;
    } else if constexpr (DerivOrder == 2) {
      analytic_func[0] = square(t);
      analytic_func[1] = 2.0 * t;
      analytic_func[2] = 2.0;
    } else if constexpr (DerivOrder == 3) {
      analytic_func[0] = cube(t);
      analytic_func[1] = 3.0 * square(t);
      analytic_func[2] = 6.0 * t;
      analytic_func[3] = 6.0;
    }

    // update exponential averager
    averager.update(t, {analytic_func[0]}, {damping_time});
    // compare values once averager has sufficient data
    if (averager(t)) {
      // update the weight and \alpha
      avg_weight = (tau_m + avg_weight) / denom;
      const double alpha = tau_m / avg_weight / denom;

      // do simple average of time, analytic func and its analytic derivatives
      avg_time = alpha * t + (1.0 - alpha) * avg_time;
      for (size_t i = 0; i < DerivOrder + 1; i++) {
        gsl::at(avg_values, i) =
            alpha * analytic_func[i] + (1.0 - alpha) * gsl::at(avg_values, i);
      }
      auto result = averager(t).value();

      // check that the effective times agree with the averaged time
      CHECK(approx(averager.average_time(t)) == avg_time);
      // check function value with stricter approx
      CHECK(approx(result[0][0]) == avg_values[0][0]);
      // check derivatives
      // Again, this check is slightly looser than the others due to
      // numerical differentiation (see comment in Averager.Linear test)
      CHECK_ITERABLE_CUSTOM_APPROX(result, avg_values, custom_approx);
    } else {
      avg_time = t;
      for (size_t i = 0; i < DerivOrder + 1; i++) {
        gsl::at(avg_values, i) = analytic_func[i];
      }
    }

    t += tau_m;
  }
}

template <size_t DerivOrder>
void test_functionality() {
  INFO("Test functionality.");
  double t = 0.0;
  const double dt = 0.1;

  Averager<DerivOrder> averager(0.5, false);

  // test the validity of data functionality
  // data not valid yet
  CHECK_FALSE(static_cast<bool>(averager(t)));
  // first update
  averager.update(t, {t}, {0.1});
  // data not valid yet
  CHECK_FALSE(static_cast<bool>(averager(t)));
  t += dt;
  // second update
  averager.update(t, {t}, {0.1});
  // different deriv orders need different number of updates for data to become
  // valid
  if constexpr (DerivOrder > 1) {
    // data not valid yet
    CHECK_FALSE(static_cast<bool>(averager(t)));
    t += dt;
    // third update
    averager.update(t, {t}, {0.1});
  }
  if constexpr (DerivOrder > 2) {
    // data not valid yet
    CHECK_FALSE(static_cast<bool>(averager(t)));
    t += dt;
    // fourth update
    averager.update(t, {t}, {0.1});
  }
  // data should be valid now
  CHECK(static_cast<bool>(averager(t)));
  CHECK(averager(t).value()[0][0] == t);
  t += dt;
  // data should currently be invalid since there was no update at this new `t`
  CHECK_FALSE(static_cast<bool>(averager(t)));
  // last updated time should then be the time before this step
  CHECK(approx(averager.last_time_updated()) == (t - dt));

  // test the clear() function:
  // update averager again, so that data is valid
  averager.update(t, {t}, {0.1});
  CHECK(static_cast<bool>(averager(t)));
  // clear the averager, which should make the data no longer valid
  averager.clear();
  CHECK_FALSE(static_cast<bool>(averager(t)));
}
}  // namespace

// [[OutputRegex, at or before the last time]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.BadUpdateTwice",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  Averager<2> averager(0.5, false);

  averager.update(0.5, {0.0}, {0.1});
  averager.update(0.5, {0.0}, {0.1});
}

// [[OutputRegex, at or before the last time]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.BadUpdatePast",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  Averager<2> averager(0.5, false);

  averager.update(0.5, {0.0}, {0.1});
  averager.update(0.3, {0.0}, {0.1});
}

// [[OutputRegex, The number of supplied timescales \(1\) does not match]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.WrongSizeTimescales",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();

  double t = 0.0;
  constexpr size_t deriv_order = 2;

  Averager<deriv_order> averager(1.0, false);
  averager.update(t, {{0.2, 0.3}}, {0.1});
}

// [[OutputRegex, The number of components in the raw_q provided \(2\) does]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.WrongSizeQProvided",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();

  double t = 0.0;
  constexpr size_t deriv_order = 2;

  Averager<deriv_order> averager(1.0, false);

  averager.update(t, {0.1}, {0.1});
  averager.update(t, {{0.2, 0.3}}, {0.1});
}

// [[OutputRegex, must be positive]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.BadAvgTimescale",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  Averager<2> averager(0.0, true);
}

// [[OutputRegex, The time history has not been updated yet]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.BadCallToLastTimeUpdated",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  Averager<2> averager(1.0, true);
  averager.last_time_updated();
}

// [[OutputRegex, Cannot return averaged values because the averager does not]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Averager.BadCallToAverageTime",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  Averager<2> averager(1.0, true);
  averager.average_time(0.0);
}

namespace {
template <size_t DerivOrder>
void test_equality_and_serialization() {
  INFO("Test equality.");
  Averager<DerivOrder> averager1(1.0, true);
  Averager<DerivOrder> averager2(1.0, true);
  Averager<DerivOrder> averager3(0.5, false);

  CHECK(averager1 == averager2);
  CHECK(averager1 != averager3);
  CHECK_FALSE(averager1 == averager3);

  averager2.update(0.1, {0.2}, {0.3});
  CHECK_FALSE(averager1 == averager2);

  CHECK(serialize_and_deserialize(averager1) == averager1);
}

template <size_t DerivOrder>
void test_create_from_options() {
  INFO("Test options.");
  Averager<DerivOrder> averager(0.5, true);
  auto averager2 = TestHelpers::test_creation<Averager<DerivOrder>>(
      "AverageTimescaleFraction: 0.5\n"
      "Average0thDeriv: True");

  CHECK(averager == averager2);
}

template <size_t DerivOrder>
void test_move() {
  INFO("Test move.");
  Averager<DerivOrder> averager(0.25, false);
  static_assert(std::is_move_constructible<Averager<DerivOrder>>::value,
                "Averager is not nothrow move constructible");
  static_assert(std::is_move_assignable<Averager<DerivOrder>>::value,
                "Averager is not nothrow move assignable");
  // update with junk data
  averager.update(0.3, {0.1}, {0.1});
  averager.update(0.5, {0.2}, {0.1});
  averager.update(0.7, {0.4}, {0.1});
  averager.update(0.9, {0.6}, {0.1});
  // get correct values for comparison
  auto last_time = averager.last_time_updated();
  auto avg_time = averager.average_time(0.9);
  auto avg_q = averager.using_average_0th_deriv_of_q();
  auto avg_values = averager(0.9).value();
  // test move constructor
  auto new_averager(std::move(averager));
  // check moved values against stored values
  CHECK(last_time == new_averager.last_time_updated());
  CHECK(avg_time == new_averager.average_time(0.9));
  CHECK(avg_q == new_averager.using_average_0th_deriv_of_q());
  CHECK(avg_values[0][0] == new_averager(0.9).value()[0][0]);
  // test move assignment
  Averager<DerivOrder> new_averager2(0.1, true);
  new_averager2 = std::move(new_averager);
  // check moved values against stored values
  CHECK(last_time == new_averager2.last_time_updated());
  CHECK(avg_time == new_averager2.average_time(0.9));
  CHECK(avg_q == new_averager2.using_average_0th_deriv_of_q());
  CHECK(avg_values[0][0] == new_averager2(0.9).value()[0][0]);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Averager", "[ControlSystem][Unit]") {
  {
    INFO("Deriv Order 1")
    test_linear<1>();
    test_semianalytic<1>();
    test_functionality<1>();
    test_equality_and_serialization<1>();
    test_create_from_options<1>();
    test_move<1>();
  }

  {
    INFO("Deriv Order 2")
    test_linear<2>();
    test_semianalytic<2>();
    test_functionality<2>();
    test_equality_and_serialization<2>();
    test_create_from_options<2>();
    test_move<2>();
  }

  {
    INFO("Deriv Order 3")
    test_linear<3>();
    test_semianalytic<3>();
    test_functionality<3>();
    test_equality_and_serialization<3>();
    test_create_from_options<3>();
    test_move<3>();
  }
}
