// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>

#include "ErrorHandling/Error.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/Time.hpp"
#include "tests/Unit/TestCreation.hpp"

class StepController;

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
