// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <limits>

#include "Time/Slab.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/StepControllers/FullSlab.hpp"
#include "Time/StepControllers/SimpleTimes.hpp"
#include "Time/StepControllers/SplitRemaining.hpp"
#include "Time/Time.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Time.StepControllers.BinaryFraction", "[Unit][Time]") {
  const StepControllers::BinaryFraction bf{};
  const Slab slab(1., 4.);
  CHECK(bf.choose_step(slab.start(), 4.) == slab.duration());
  CHECK(bf.choose_step(slab.start(), 10.) == slab.duration());
  CHECK(bf.choose_step(slab.start(), 2.) == slab.duration() / 2);
  CHECK(bf.choose_step(slab.start(), 1.4) == slab.duration() / 4);
  CHECK(bf.choose_step(slab.start() + slab.duration() / 4, 2.) ==
        slab.duration() / 4);

  CHECK(bf.choose_step(slab.end(), -2.) == -slab.duration() / 2);

  CHECK(bf.choose_step(slab.start(), std::numeric_limits<double>::infinity())
        == slab.duration());
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.BinaryFraction.Factory",
                  "[Unit][Time]") {
  test_factory_creation<StepController>("  BinaryFraction");
}

// [[OutputRegex, Not at a binary-fraction time within slab]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Time.StepControllers.BinaryFraction.NonBinary", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const StepControllers::BinaryFraction bf{};
  const Slab slab(1., 4.);
  bf.choose_step(slab.start() + slab.duration() / 3, 1.);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.FullSlab", "[Unit][Time]") {
  const StepControllers::FullSlab fs{};
  const Slab slab(1., 4.);
  CHECK(fs.choose_step(slab.start(), 4.) == slab.duration());
  CHECK(fs.choose_step(slab.start(), 10.) == slab.duration());
  CHECK(fs.choose_step(slab.start(), 2.) == slab.duration());
  CHECK(fs.choose_step(slab.start(), 1.4) == slab.duration());

  CHECK(fs.choose_step(slab.end(), -2.) == -slab.duration());

  CHECK(fs.choose_step(slab.start(), std::numeric_limits<double>::infinity())
        == slab.duration());
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.FullSlab.Factory",
                  "[Unit][Time]") {
  test_factory_creation<StepController>("  FullSlab");
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.SimpleTimes", "[Unit][Time]") {
  const StepControllers::SimpleTimes st{};
  const Slab slab(1., 4.);
  CHECK(st.choose_step(slab.start(), 4.) == slab.duration());
  CHECK(st.choose_step(slab.start(), 10.) == slab.duration());
  CHECK(st.choose_step(slab.start(), 2.) == slab.duration() / 2);
  CHECK(st.choose_step(slab.start(), 1.4) == slab.duration() / 3);

  using rational_t = TimeDelta::rational_t;
  // Limited by small step prevention
  CHECK(st.choose_step(slab.start() + slab.duration() / 4, 2.) ==
        slab.duration() * (rational_t(2, 3) - rational_t(1, 4)));
  // Limited by extra step prevention
  CHECK(st.choose_step(slab.start() + slab.duration() / 4, 0.48) ==
        slab.duration() * (rational_t(2, 5) - rational_t(1, 4)));

  CHECK(st.choose_step(slab.end(), -2.) == -slab.duration() / 2);

  CHECK(st.choose_step(slab.start(), std::numeric_limits<double>::infinity())
        == slab.duration());
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.SimpleTimes.Factory",
                  "[Unit][Time]") {
  test_factory_creation<StepController>("  SimpleTimes");
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.SplitRemaining", "[Unit][Time]") {
  const StepControllers::SplitRemaining sr{};
  const Slab slab(1., 4.);
  CHECK(sr.choose_step(slab.start(), 4.) == slab.duration());
  CHECK(sr.choose_step(slab.start(), 10.) == slab.duration());
  CHECK(sr.choose_step(slab.start(), 2.) == slab.duration() / 2);
  CHECK(sr.choose_step(slab.start(), 1.4) == slab.duration() / 3);
  CHECK(sr.choose_step(slab.start() + slab.duration() / 4, 2.) ==
        slab.duration() * 3 / 8);
  CHECK(sr.choose_step(slab.start() + slab.duration() / 3, 0.7) ==
        slab.duration() * 2 / 9);

  CHECK(sr.choose_step(slab.end(), -2.) == -slab.duration() / 2);

  CHECK(sr.choose_step(slab.start(), std::numeric_limits<double>::infinity())
        == slab.duration());
}

SPECTRE_TEST_CASE("Unit.Time.StepControllers.SplitRemaining.Factory",
                  "[Unit][Time]") {
  test_factory_creation<StepController>("  SplitRemaining");
}
