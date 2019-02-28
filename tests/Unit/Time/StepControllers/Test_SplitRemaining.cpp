// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <memory>

#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/SplitRemaining.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Time.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

SPECTRE_TEST_CASE("Unit.Time.StepControllers.SplitRemaining", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepController>();
  const auto check = [](const auto& sr) noexcept {
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

    CHECK(
        sr.choose_step(slab.start(), std::numeric_limits<double>::infinity()) ==
        slab.duration());
  };
  check(StepControllers::SplitRemaining{});
  check(*serialize_and_deserialize(
      test_factory_creation<StepController>("  SplitRemaining")));
}
