// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <functional>
#include <initializer_list>  // IWYU pragma: keep
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <cmath>  // for abs

// IWYU pragma: no_include "Utilities/Rational.hpp"

SPECTRE_TEST_CASE("Unit.Time.Time", "[Unit][Time]") {
  using rational_t = Time::rational_t;

  const double tstart_d = 0.68138945475734402635;
  const double tend_d = 76.34481744714527451379;
  // Make sure we're using values that will trigger rounding errors.
  CHECK_FALSE(tstart_d + (tend_d - tstart_d) == tend_d);

  const Slab slab(tstart_d, tend_d);
  CHECK(Time(slab, 0).slab() == slab);
  CHECK(Time(slab, rational_t(3, 5)).fraction() == rational_t(3, 5));

  CHECK(Time(slab, 0).value() == tstart_d);
  CHECK(Time(slab, 1).value() == tend_d);
  CHECK(Time(slab, rational_t(3, 5)).value() ==
        approx(2. / 5. * tstart_d + 3. / 5. * tend_d));

  CHECK(Time(slab, 0).is_at_slab_start());
  CHECK_FALSE(Time(slab, rational_t(1, 2)).is_at_slab_start());
  CHECK_FALSE(Time(slab, 1).is_at_slab_start());
  CHECK_FALSE(Time(slab, 0).is_at_slab_end());
  CHECK_FALSE(Time(slab, rational_t(1, 2)).is_at_slab_end());
  CHECK(Time(slab, 1).is_at_slab_end());
  CHECK(Time(slab, 0).is_at_slab_boundary());
  CHECK_FALSE(Time(slab, rational_t(1, 2)).is_at_slab_boundary());
  CHECK(Time(slab, 1).is_at_slab_boundary());

  check_cmp(Time(slab, 0), Time(slab, 1));
  check_cmp(Time(slab, 0), Time(slab, rational_t(3, 5)));
  check_cmp(Time(slab, rational_t(3, 5)), Time(slab, 1));
  check_cmp(Time(slab, rational_t(2, 5)), Time(slab, rational_t(3, 5)));
  check_cmp(Time(slab, 1), Time(slab.advance(), 1));
  check_cmp(Time(slab, 0), Time(slab.advance(), 0));
  check_cmp(Time(slab, rational_t(3, 5)),
            Time(slab.advance(), rational_t(2, 5)));

  CHECK(Time(slab, rational_t(2, 3)).with_slab(slab) ==
        Time(slab, rational_t(2, 3)));

  {
    // Slab boundary stuff
    const double other_duration = 2. * slab.duration().value();

    const Time a2(slab, 1);
    const Time b2 = a2.with_slab(slab.advance());
    CHECK(b2.slab() == slab.advance());
    CHECK(b2.fraction() == 0);
    const Time c2 = a2.with_slab(slab.with_duration_to_end(other_duration));
    CHECK(c2.slab() == slab.with_duration_to_end(other_duration));
    CHECK(c2.fraction() == 1);
  }

  CHECK(Time(slab, 0).value() == Time(slab.retreat(), 1).value());
  CHECK(Time(slab, 1).value() == Time(slab.advance(), 0).value());

  CHECK(Time(slab, rational_t(3, 5)) - Time(slab, rational_t(1, 5)) ==
        TimeDelta(slab, rational_t(2, 5)));

  CHECK_OP(Time(slab, rational_t(1, 5)), +, TimeDelta(slab, rational_t(2, 5)),
           Time(slab, rational_t(3, 5)));
  CHECK_OP(Time(slab, rational_t(3, 5)), -, TimeDelta(slab, rational_t(2, 5)),
           Time(slab, rational_t(1, 5)));

  // Slab boundary arithmetic
  CHECK(Time(slab.advance(), 0) - Time(slab, rational_t(2, 3)) ==
        TimeDelta(slab, rational_t(1, 3)));
  CHECK(Time(slab, rational_t(2, 3)) - Time(slab.advance(), 0) ==
        TimeDelta(slab, -rational_t(1, 3)));
  CHECK(Time(slab, 1) - Time(slab.advance(), rational_t(2, 3)) ==
        TimeDelta(slab.advance(), -rational_t(2, 3)));
  CHECK(Time(slab.advance(), rational_t(2, 3)) - Time(slab, 1) ==
        TimeDelta(slab.advance(), rational_t(2, 3)));

  CHECK_OP(Time(slab, 1), +, TimeDelta(slab.advance(), rational_t(1, 3)),
           Time(slab.advance(), rational_t(1, 3)));
  CHECK_OP(Time(slab, 1), -, TimeDelta(slab.advance(), -rational_t(1, 3)),
           Time(slab.advance(), rational_t(1, 3)));
  CHECK_OP(Time(slab, 0), +, TimeDelta(slab.retreat(), -rational_t(1, 3)),
           Time(slab.retreat(), rational_t(2, 3)));
  CHECK_OP(Time(slab, 0), -, TimeDelta(slab.retreat(), rational_t(1, 3)),
           Time(slab.retreat(), rational_t(2, 3)));

  // Hashing
  std::hash<Time> h;
  CHECK(h(Slab(0, 1).start()) == h(Slab(0, 2).start()));
  CHECK(h(Slab(0, 1).start()) != h(Slab(0, 1).end()));
  CHECK(h(Slab(0, 1).start()) == h(Slab(-2, 0).end()));
  CHECK(h(Slab(-1, 0).end()) == h(Slab(-2, 0).end()));
  CHECK(h(Time(Slab(0, 1), rational_t(1, 2))) !=
        h(Time(Slab(0, 2), rational_t(1, 3))));

  // Output
  const std::string slab_str = get_output(slab);
  CHECK(get_output(Time(slab, 0)) == slab_str + ":0/1");
  CHECK(get_output(Time(slab, 1)) == slab_str + ":1/1");
  CHECK(get_output(Time(slab, rational_t(3, 5))) == slab_str + ":3/5");

  const auto check_structural_compare =
      [](const Time& a, const Time& b) noexcept {
    Time::StructuralCompare sc;
    CHECK_FALSE(sc(a, a));
    CHECK_FALSE(sc(b, b));
    CHECK(sc(a, b) != sc(b, a));
  };
  check_structural_compare(Time(slab, rational_t(1, 3)),
                           Time(slab, rational_t(2, 3)));
  check_structural_compare(Time(slab, rational_t(1, 3)),
                           Time(slab, rational_t(1, 2)));
  check_structural_compare(Time(slab, rational_t(0, 1)),
                           Time(slab.advance(), rational_t(0, 1)));
}

SPECTRE_TEST_CASE("Unit.Time.Time_slab_comparison", "[Unit][Time]") {
  const double tstart_d = 0.68138945475734402635;
  const double tend_d = 76.34481744714527451379;
  // Make sure we're using values that will trigger rounding errors.
  CHECK_FALSE(tstart_d + (tend_d - tstart_d) == tend_d);

  const Slab slab(tstart_d, tend_d);

  const double other_duration = 2. * slab.duration().value();

  const Time a(slab, 0);
  const Time b = a.with_slab(slab.retreat());
  CHECK(b.slab() == slab.retreat());
  CHECK(b.fraction() == 1);
  const Time c = a.with_slab(slab.with_duration_from_start(other_duration));
  CHECK(c.slab() == slab.with_duration_from_start(other_duration));
  CHECK(c.fraction() == 0);
  const Time d =
      a.with_slab(slab.retreat().with_duration_to_end(other_duration));
  CHECK(d.slab() == slab.retreat().with_duration_to_end(other_duration));
  CHECK(d.fraction() == 1);

  const std::array<Time, 4> equal_times{{a, b, c, d}};
  for (const auto& t1 : equal_times) {
    for (const auto& t2 : equal_times) {
      CHECK(t1 == t2);
      CHECK_FALSE(t1 != t2);
      CHECK_FALSE(t1 < t2);
      CHECK_FALSE(t1 > t2);
      CHECK(t1 <= t2);
      CHECK(t1 >= t2);
    }
  }

  for (const auto& t1 : {a, c}) {
    for (const auto& t2 : {b.slab().start(), d.slab().start()}) {
      CHECK_FALSE(t1 == t2);
      CHECK_FALSE(t2 == t1);
      CHECK(t1 != t2);
      CHECK(t2 != t1);
      CHECK_FALSE(t1 < t2);
      CHECK_FALSE(t2 > t1);
      CHECK(t1 > t2);
      CHECK(t2 < t1);
      CHECK_FALSE(t1 <= t2);
      CHECK_FALSE(t2 >= t1);
      CHECK(t1 >= t2);
      CHECK(t2 <= t1);
    }
  }
  for (const auto& t1 : {b, d}) {
    for (const auto& t2 : {a.slab().end(), c.slab().end()}) {
      CHECK_FALSE(t1 == t2);
      CHECK_FALSE(t2 == t1);
      CHECK(t1 != t2);
      CHECK(t2 != t1);
      CHECK(t1 < t2);
      CHECK(t2 > t1);
      CHECK_FALSE(t1 > t2);
      CHECK_FALSE(t2 < t1);
      CHECK(t1 <= t2);
      CHECK(t2 >= t1);
      CHECK_FALSE(t1 >= t2);
      CHECK_FALSE(t2 <= t1);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Time.Time.serialization",
                  "[Unit][Time][Serialization]") {
  const Slab slab(1.7, 2.4);
  const Time time = slab.start() + slab.duration() * 3 / 5;
  test_serialization(time);
}

SPECTRE_TEST_CASE("Unit.Time.TimeDelta", "[Unit][Time]") {
  using rational_t = TimeDelta::rational_t;

  const double tstart_d = 0.68138945475734402635;
  const double tend_d = 76.34481744714527451379;
  const double length_d = tend_d - tstart_d;
  // Make sure we're using values that will trigger rounding errors.
  CHECK_FALSE(tstart_d + length_d == tend_d);

  const Slab slab(tstart_d, tend_d);
  CHECK(TimeDelta(slab, 0).slab() == slab);
  CHECK(TimeDelta(slab, rational_t(3, 5)).fraction() == rational_t(3, 5));

  CHECK(TimeDelta(slab, 0).value() == 0);
  CHECK(TimeDelta(slab, 1).value() == approx(length_d));
  CHECK(TimeDelta(slab, rational_t(1, 5)).value() == approx(length_d / 5));
  // TimeDeltas can be "out of range".
  CHECK(TimeDelta(slab, -rational_t(1, 5)).value() == approx(-length_d / 5));
  CHECK(TimeDelta(slab, 2).value() == approx(2 * length_d));

  CHECK(TimeDelta(slab, rational_t(1, 2)).is_positive());
  CHECK_FALSE(TimeDelta(slab, -rational_t(1, 2)).is_positive());
  CHECK_FALSE(TimeDelta(slab, 0).is_positive());

  {
    const Slab slab2(10., 14.);
    const TimeDelta delta(slab, rational_t(1, 3));
    const TimeDelta delta2 = delta.with_slab(slab2);
    CHECK(delta2.slab() == slab2);
    CHECK(delta2.fraction() == delta.fraction());
  }

  check_cmp(TimeDelta(slab, rational_t(2, 5)),
            TimeDelta(slab, rational_t(3, 5)));

  CHECK(+TimeDelta(slab, rational_t(3, 5)) ==
        TimeDelta(slab, rational_t(3, 5)));

  CHECK(-TimeDelta(slab, rational_t(3, 5)) ==
        TimeDelta(slab, -rational_t(3, 5)));

  CHECK(TimeDelta(slab, rational_t(3, 5)) / TimeDelta(slab, rational_t(3, 5)) ==
        1.);
  CHECK(TimeDelta(slab, rational_t(3, 5)) / TimeDelta(slab, rational_t(2, 5)) ==
        approx(1.5));
  // This is the only operation that is valid arbitrarily cross-slab.
  CHECK(TimeDelta(slab.advance().with_duration_from_start(2.345),
                  rational_t(2, 3)) /
            TimeDelta(slab, rational_t(4, 5)) ==
        approx(2.345 * 2. / 3. / (length_d * 4. / 5.)));

  CHECK(TimeDelta(slab, rational_t(2, 5)) + Time(slab, rational_t(1, 5)) ==
        Time(slab, rational_t(3, 5)));

  CHECK_OP(TimeDelta(slab, rational_t(1, 5)), +,
           TimeDelta(slab, rational_t(2, 5)),
           TimeDelta(slab, rational_t(3, 5)));
  CHECK_OP(TimeDelta(slab, rational_t(1, 5)), -,
           TimeDelta(slab, rational_t(2, 5)),
           TimeDelta(slab, -rational_t(1, 5)));
  CHECK_OP(TimeDelta(slab, rational_t(1, 5)), *, rational_t(2, 5),
           TimeDelta(slab, rational_t(2, 25)));
  CHECK_OP(TimeDelta(slab, rational_t(1, 5)), /, rational_t(2, 5),
           TimeDelta(slab, rational_t(1, 2)));

  CHECK(rational_t(2, 5) * TimeDelta(slab, rational_t(1, 5)) ==
        TimeDelta(slab, rational_t(2, 25)));

  CHECK(abs(TimeDelta(slab, rational_t(1, 5))) ==
        TimeDelta(slab, rational_t(1, 5)));
  CHECK(abs(TimeDelta(slab, -rational_t(1, 5))) ==
        TimeDelta(slab, rational_t(1, 5)));

  // Slab boundary arithmetic
  CHECK(TimeDelta(slab.advance(), rational_t(1, 3)) + Time(slab, 1) ==
        Time(slab.advance(), rational_t(1, 3)));
  CHECK(TimeDelta(slab.retreat(), -rational_t(1, 3)) + Time(slab, 0) ==
        Time(slab.retreat(), rational_t(2, 3)));

  // Output
  const std::string slab_str = get_output(slab);
  CHECK(get_output(TimeDelta(slab, 0)) == slab_str + ":0/1");
  CHECK(get_output(TimeDelta(slab, 1)) == slab_str + ":1/1");
  CHECK(get_output(TimeDelta(slab, rational_t(3, 5))) == slab_str + ":3/5");
  CHECK(get_output(TimeDelta(slab, rational_t(3, -5))) == slab_str + ":-3/5");
}

SPECTRE_TEST_CASE("Unit.Time.TimeDelta.serialization",
                  "[Unit][Time][Serialization]") {
  const Slab slab(1.7, 2.4);
  const TimeDelta dt = slab.duration() * 3 / 5;
  test_serialization(dt);
}

// Failure tests
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wunused-comparison"
#endif

/// [example_of_error_test]
// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.Init.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), -1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
/// [example_of_error_test]

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.Init.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), Time::rational_t(1, 2)).with_slab(Slab(1., 2.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), Time::rational_t(1, 2)).with_slab(Slab(-1., 0.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), Time::rational_t(1, 2)).with_slab(Slab(0., 2.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.3", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0).with_slab(Slab(1., 2.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.4", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0).with_slab(Slab(-1., 1.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.5", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 1).with_slab(Slab(-1., 0.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.with_slab.6", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 1).with_slab(Slab(0., 2.));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.overlap", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 1) < Time(Slab(0., 2.), 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't subtract times from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.subtraction.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) - Time(Slab(2., 3.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't subtract times from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.subtraction.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 1) - Time(Slab(-1., 0.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't subtract times from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.subtraction.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(-1., 0.), 0) - Time(Slab(0., 1.), 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) + TimeDelta(Slab(0., 1.), 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) += TimeDelta(Slab(0., 1.), 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) + TimeDelta(Slab(0., 1.), -2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.3", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) += TimeDelta(Slab(0., 1.), -2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.4", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) - TimeDelta(Slab(0., 1.), 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.5", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) -= TimeDelta(Slab(0., 1.), 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.6", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) - TimeDelta(Slab(0., 1.), -2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.7", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) -= TimeDelta(Slab(0., 1.), -2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.8", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) + TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.9", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) += TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.10", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) - TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Time.add_delta.11", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Time(Slab(0., 1.), 0) -= TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't check cross-slab TimeDelta inequalities]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.inequalities.lt",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) < TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't check cross-slab TimeDelta inequalities]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.inequalities.gt",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) > TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't check cross-slab TimeDelta inequalities]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.inequalities.le",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) <= TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't check cross-slab TimeDelta inequalities]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.inequalities.ge",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) >= TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_time.0",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 2) + Time(Slab(0., 1.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Out of range slab fraction]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_time.1",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), -2) + Time(Slab(0., 1.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't move .* to slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_time.2",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(1., 2.), 0) + Time(Slab(0., 1.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't add TimeDeltas from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_delta.0",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) += TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't add TimeDeltas from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_delta.1",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) + TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't subtract TimeDeltas from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_delta.2",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) -= TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't subtract TimeDeltas from different slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.TimeDelta.add_delta.3",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TimeDelta(Slab(0., 1.), 0) - TimeDelta(Slab(1., 2.), 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
