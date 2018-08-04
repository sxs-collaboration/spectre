// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Time.Slab", "[Unit][Time]") {
  const double tstart_d = 0.68138945475734402635;
  const double tend_d = 76.34481744714527451379;
  // Make sure we're using values that will trigger rounding errors.
  CHECK_FALSE(tstart_d + (tend_d - tstart_d) == tend_d);
  const Slab slab(tstart_d, tend_d);

  // Arbitrary
  const double tend2_d = tend_d + 1.234;
  // Arbitrary duration, not related to above slabs.
  const double duration_d = tend2_d;

  {
    CHECK(slab.start().value() == tstart_d);
    CHECK(slab.end().value() == tend_d);
    CHECK(slab.duration().value() == approx(tend_d - tstart_d));
    CHECK(slab.end() - slab.start() == slab.duration());
  }

  {
    const Slab slab2 = Slab::with_duration_from_start(tstart_d, duration_d);
    CHECK(slab2.start().value() == tstart_d);
    CHECK(slab2.duration().value() == approx(duration_d));
  }

  {
    const Slab slab2 = Slab::with_duration_to_end(tend_d, duration_d);
    CHECK(slab2.end().value() == tend_d);
    CHECK(slab2.duration().value() == approx(duration_d));
  }

  {
    const Slab next = slab.advance();
    CHECK(next.start() == slab.end());
    CHECK(next.duration().value() == approx(slab.duration().value()));
    const Slab prev = slab.retreat();
    CHECK(prev.end() == slab.start());
    CHECK(prev.duration().value() == approx(slab.duration().value()));

    CHECK(slab.advance_towards(slab.duration()) == next);
    CHECK(slab.advance_towards(-slab.duration()) == prev);
  }

  {
    const Slab front = slab.with_duration_from_start(duration_d);
    CHECK(front.start().value() == slab.start().value());
    CHECK(front.duration().value() == approx(duration_d));
    const Slab back = slab.with_duration_to_end(duration_d);
    CHECK(back.end().value() == slab.end().value());
    CHECK(back.duration().value() == approx(duration_d));
  }

  {
    CHECK(slab.is_followed_by(slab.advance().with_duration_from_start(3)));
    CHECK(slab.is_preceeded_by(slab.retreat().with_duration_to_end(3)));
  }

  CHECK(slab == slab);
  CHECK_FALSE(slab == Slab(tstart_d, tend2_d));
  CHECK(slab != Slab(tstart_d, tend2_d));
  CHECK(slab != Slab(tend2_d / 2., tend_d));
  CHECK(slab != Slab(tend_d, tend2_d));

  check_cmp(Slab(1, 2), Slab(3, 4));
  check_cmp(Slab(1, 2), Slab(2, 4));

  // Hashing
  std::hash<Slab> h;
  CHECK(h(Slab(0, 1)) != h(Slab(0, 2)));
  CHECK(h(Slab(0, 1)) != h(Slab(-1, 0)));
  CHECK(h(Slab(0, 1)) != h(Slab(-1, 1)));

  // Output
  CHECK(get_output(Slab(0.5, 1.5)) == "Slab[0.5,1.5]");
}

SPECTRE_TEST_CASE("Unit.Time.Slab.serialization",
                  "[Unit][Time][Serialization]") {
  const Slab slab(1.7, 2.4);
  test_serialization(slab);
}

// Failure tests
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wunused-comparison"
#endif

// [[OutputRegex, Backwards Slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.Backwards.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(1., 0.);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Backwards Slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.Backwards.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab::with_duration_from_start(0., -1.);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Backwards Slab]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.Backwards.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab::with_duration_to_end(0., -1.);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Can't advance along a zero time vector]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.Advance0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  slab.advance_towards(0 * slab.duration());
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) < Slab(0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) < Slab(0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) < Slab(-0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less.3", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) < Slab(-0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) > Slab(0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) > Slab(0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) > Slab(-0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater.3", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) > Slab(-0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less_equal.0", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) <= Slab(0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less_equal.1", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) <= Slab(0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less_equal.2", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) <= Slab(-0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.less_equal.3", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) <= Slab(-0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater_equal.0",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) >= Slab(0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater_equal.1",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) >= Slab(0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater_equal.2",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) >= Slab(-0.1, 0.9);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot compare overlapping slabs]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.Slab.greater_equal.3",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Slab(0., 1.) >= Slab(-0.1, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
