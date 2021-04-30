// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <memory>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

SPECTRE_TEST_CASE("Unit.Time.StepControllers.BinaryFraction", "[Unit][Time]") {
  Parallel::register_classes_with_charm<StepControllers::BinaryFraction>();
  const auto check = [](const auto& bf) noexcept {
    const Slab slab(1., 4.);
    CHECK(bf.choose_step(slab.start(), 4.) == slab.duration());
    CHECK(bf.choose_step(slab.start(), 10.) == slab.duration());
    CHECK(bf.choose_step(slab.start(), 2.) == slab.duration() / 2);
    CHECK(bf.choose_step(slab.start(), 1.4) == slab.duration() / 4);
    CHECK(bf.choose_step(slab.start() + slab.duration() / 4, 2.) ==
          slab.duration() / 4);

    CHECK(bf.choose_step(slab.end(), -2.) == -slab.duration() / 2);

    CHECK(
        bf.choose_step(slab.start(), std::numeric_limits<double>::infinity()) ==
        slab.duration());
  };
  check(StepControllers::BinaryFraction{});
  check(*serialize_and_deserialize(
      TestHelpers::test_factory_creation<
          StepController, StepControllers::BinaryFraction>("BinaryFraction")));
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
