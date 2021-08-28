// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
void test_increase_or_decrease() {
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);

  const DataVector timescale{1.0};
  CHECK(tst.current_timescale() == timescale);

  // Check the suggested timescale function
  // (1) timescale < min_timescale
  double suggested_tscale = min_timescale * 0.5;
  tst.set_timescale_if_in_allowable_range(suggested_tscale);
  CHECK(tst.current_timescale() ==
        make_with_value<DataVector>(timescale, min_timescale));
  // (2) min_timescale < timescale < max_timescale
  suggested_tscale = 0.5 * (min_timescale + max_timescale);
  tst.set_timescale_if_in_allowable_range(suggested_tscale);
  CHECK(tst.current_timescale() ==
        make_with_value<DataVector>(timescale, suggested_tscale));
  // (3) max_timescale < timescale
  suggested_tscale = max_timescale * 2.0;
  tst.set_timescale_if_in_allowable_range(suggested_tscale);
  CHECK(tst.current_timescale() ==
        make_with_value<DataVector>(timescale, max_timescale));

  // set timescale for remaining tests
  double tscale = 10.0;
  tst.set_timescale_if_in_allowable_range(tscale);
  CHECK(tst.current_timescale() ==
        make_with_value<DataVector>(timescale, tscale));

  // helper vars for greater than and less than one, used to choose
  // the correct Q for each CHECK
  const double greater_than_one = 1.1;
  const double less_than_one = 0.9;

  auto run_tests = [&](double sign_of_q) {
    // Here, situations that trigger the outer conditional related to a decrease
    // in timescale are numbered, while the nested conditional choices for each
    // associated outer case are suffixed with letters

    // (1) |Q| > decrease_timescale_threshold
    //     |\dot{Q}| <= decrease_timescale_threshold/timescale
    DataVector q{sign_of_q * greater_than_one * decrease_timescale_threshold};
    DataVector qdot{sign_of_q * less_than_one * decrease_timescale_threshold /
                    tscale};

    // (1a) Q\dot{Q} > 0
    //      |\dot{Q}| >= 0.5*|Q|/timescale
    // the error is large and growing: decrease timescale
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (1b) Q\dot{Q} <= 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // the error is not decreasing quickly enough: decrease timescale
    qdot = -less_than_one * 0.5 * q / tscale;
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (1c) Q\dot{Q} > 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // the error is large and growing quickly: decrease timescale
    qdot *= -1.0;
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (2) |Q| <= decrease_timescale_threshold
    //     |\dot{Q}| > decrease_timescale_threshold/timescale
    q = sign_of_q * less_than_one * decrease_timescale_threshold;
    qdot = sign_of_q * greater_than_one * decrease_timescale_threshold / tscale;

    // (2a) Q\dot{Q} > 0
    //      |\dot{Q}| >= 0.5*|Q|/timescale
    // the error is growing quickly: decrease timescale
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (2b) Q\dot{Q} <= 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // NOTE: the second piece is unachievable, given the conditions of (2).
    // This check is included in the do nothing test.

    // (2c) Q\dot{Q} > 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // NOTE: the second piece is unachievable, given the conditions of (2),
    // however, Q\dot{Q} > 0 is sufficient to trigger a decrease.
    // the error is growing quickly: decrease timescale
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (3) |Q| > decrease_timescale_threshold
    //     |\dot{Q}| > decrease_timescale_threshold/timescale
    q = sign_of_q * greater_than_one * decrease_timescale_threshold;
    qdot = sign_of_q * greater_than_one * decrease_timescale_threshold / tscale;

    // (3a) Q\dot{Q} > 0
    //      |\dot{Q}| >= 0.5*|Q|/timescale
    // the error is large and growing quickly: decrease timescale
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (3b) Q\dot{Q} <= 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // the error is large and not decreasing quickly enough: decrease timescale
    qdot *= -1.0;
    q = sign_of_q * greater_than_one * 2.0 * greater_than_one *
        decrease_timescale_threshold;
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (3c) Q\dot{Q} > 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // the error is large and growing quickly: decrease timescale
    qdot *= -1.0;
    tscale *= decrease_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // There is only one case which triggers an increase in the timescale
    // (4) |Q| < increase_timescale_threshold
    //     |\dot{Q}| < (increase_timescale_threshold-|Q|)/timescale
    // the error and time derivative are sufficiently small: increase timescale
    q = sign_of_q * less_than_one * increase_timescale_threshold;
    qdot = sign_of_q * less_than_one *
           (increase_timescale_threshold - fabs(q)) / tscale;
    tscale *= increase_factor;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));
  };
  // testing for Q>0 and Q<0
  run_tests(1.0);
  run_tests(-1.0);
}

void test_no_change_to_timescale() {
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);

  const DataVector timescale{1.0};
  CHECK(tst.current_timescale() == timescale);

  double tscale = 10.0;
  tst.set_timescale_if_in_allowable_range(tscale);
  CHECK(tst.current_timescale() ==
        make_with_value<DataVector>(timescale, tscale));

  // helper vars for greater than and less than one, used to choose
  // the correct Q for each CHECK
  const double greater_than_one = 1.1;
  const double less_than_one = 0.9;

  auto run_tests = [&](double sign_of_q) {
    // (1) |Q| > decrease_timescale_threshold
    //     |\dot{Q}| <= decrease_timescale_threshold/timescale
    DataVector q{sign_of_q * greater_than_one * decrease_timescale_threshold};
    DataVector qdot{-1.0 * sign_of_q * less_than_one *
                    decrease_timescale_threshold / tscale};

    // (1d) Q\dot{Q} <= 0
    //      |\dot{Q}| >= 0.5*|Q|/timescale
    // the error is large, but decreasing quickly enough: do nothing
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (2) |Q| <= decrease_timescale_threshold
    //     |\dot{Q}| > decrease_timescale_threshold/timescale
    q = sign_of_q * less_than_one * decrease_timescale_threshold;
    qdot = -1.0 * sign_of_q * greater_than_one * decrease_timescale_threshold /
           tscale;

    // (2b) Q\dot{Q} <= 0
    //      |\dot{Q}| < 0.5*|Q|/timescale
    // NOTE: the second piece is unachievable, given the conditions of (2).
    // the error is small and decreasing: do nothing
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (2d) Q\dot{Q} <= 0
    //      |\dot{Q}| >= 0.5*|Q|/timescale
    // the error is small and decreasing quickly: do nothing
    q = less_than_one * 2.0 * qdot * tscale;
    qdot *= -1.0;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (3) |Q| > decrease_timescale_threshold
    //     |\dot{Q}| > decrease_timescale_threshold/timescale
    q = sign_of_q * greater_than_one * decrease_timescale_threshold;
    qdot = -1.0 * sign_of_q * greater_than_one * decrease_timescale_threshold /
           tscale;

    // (3d) Q\dot{Q} <= 0
    //      |\dot{Q}| >= 0.5*|Q|/timescale
    // the error is large, but decreasing quickly: do nothing
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (4) |Q| < increase_timescale_threshold
    //     |\dot{Q}| >= (increase_timescale_threshold-|Q|)/timescale
    // the error and time derivative are sufficiently small: increase timescale
    q = sign_of_q * less_than_one * increase_timescale_threshold;
    qdot = sign_of_q * greater_than_one *
           (increase_timescale_threshold - fabs(q)) / tscale;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (5) |Q| >= increase_timescale_threshold
    //     |\dot{Q}| < (increase_timescale_threshold-|Q|)/timescale
    // the error and time derivative are sufficiently small: increase timescale
    q = sign_of_q * greater_than_one * increase_timescale_threshold;
    qdot = sign_of_q * less_than_one *
           (increase_timescale_threshold - fabs(q)) / tscale;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));

    // (6) |Q| >= increase_timescale_threshold
    //     |\dot{Q}| >= (increase_timescale_threshold-|Q|)/timescale
    // the error and time derivative are sufficiently small: increase timescale
    q = sign_of_q * greater_than_one * increase_timescale_threshold;
    qdot = sign_of_q * greater_than_one *
           (increase_timescale_threshold - fabs(q)) / tscale;
    tst.update_timescale({{q, qdot}});
    CHECK(tst.current_timescale() ==
          make_with_value<DataVector>(timescale, tscale));
  };
  run_tests(1.0);   // test positive Q
  run_tests(-1.0);  // test negative Q
}

void test_create_from_options() {
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const auto tst = TestHelpers::test_creation<TimescaleTuner>(
      "InitialTimescales: [1.]\n"
      "MinTimescale: 1e-3\n"
      "MaxTimescale: 10.\n"
      "DecreaseThreshold: 1e-2\n"
      "IncreaseThreshold: 1e-4\n"
      "IncreaseFactor: 1.01\n"
      "DecreaseFactor: 0.99\n");
  CHECK(tst == TimescaleTuner({1.}, max_timescale, min_timescale,
                              decrease_timescale_threshold,
                              increase_timescale_threshold, increase_factor,
                              decrease_factor));
}

void test_equality_and_serialization() {
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  TimescaleTuner tst1(
      {0.3}, max_timescale, min_timescale, decrease_timescale_threshold,
      increase_timescale_threshold, increase_factor, decrease_factor);

  TimescaleTuner tst2(
      {0.4}, max_timescale, min_timescale, decrease_timescale_threshold,
      increase_timescale_threshold, increase_factor, decrease_factor);

  CHECK(tst1 == tst1);
  CHECK(tst1 != tst2);
  CHECK(serialize_and_deserialize(tst1) == tst1);
}
}  // namespace

// [[OutputRegex, Initial timescale must be > 0]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadInitTimescale",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const std::vector<double> init_timescale{0.0};
  TimescaleTuner tst(init_timescale, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, must satisfy 0 < decrease_factor <= 1]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadDecFactor0",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const double increase_factor = 1.1;
  const double decrease_factor = -0.99;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, must satisfy 0 < decrease_factor <= 1]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadDecFactor1",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const double increase_factor = 1.1;
  const double decrease_factor = 1.01;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, must be >= 1]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadIncFactor",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const double increase_factor = 0.99;
  const double decrease_factor = 0.8;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, must be > 0]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadMinTimescale",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;

  const double max_timescale = 10.0;
  const double min_timescale = 0.0;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, must be > than the specified minimum timescale]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadMaxTimescale",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;

  const double max_timescale = 1.0e-4;
  const double min_timescale = 1.0e-3;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, The specified increase-timescale threshold]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadIncreaseThreshold",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 0.0;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, must be > than the specified increase-timescale threshold]]
SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.BadDecreaseThreshold",
                  "[ControlSystem][Unit]") {
  ERROR_TEST();
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const double decrease_timescale_threshold = 1.0e-4;
  const double increase_timescale_threshold = 1.0e-3;

  TimescaleTuner tst({1.0}, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);
}

// [[OutputRegex, One or both of the number of components in q_and_dtq]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner.SizeMismatch",
                               "[ControlSystem][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const double decrease_timescale_threshold = 1.0e-2;
  const double increase_timescale_threshold = 1.0e-4;
  const double increase_factor = 1.01;
  const double decrease_factor = 0.99;
  const double max_timescale = 10.0;
  const double min_timescale = 1.0e-3;

  const std::vector<double> init_timescale{{1.0, 2.0}};
  TimescaleTuner tst(init_timescale, max_timescale, min_timescale,
                     decrease_timescale_threshold, increase_timescale_threshold,
                     increase_factor, decrease_factor);

  const std::array<DataVector, 2> qs{{{2.0}, {3.0}}};
  tst.update_timescale(qs);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.ControlSystem.TimescaleTuner",
                  "[ControlSystem][Unit]") {
  test_increase_or_decrease();
  test_no_change_to_timescale();
  test_create_from_options();
  test_equality_and_serialization();
}
