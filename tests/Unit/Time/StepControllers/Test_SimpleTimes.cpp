// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <memory>

#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/SimpleTimes.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Time.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"
// IWYU pragma: no_include "Utilities/Rational.hpp"

SPECTRE_TEST_CASE("Unit.Time.StepControllers.SimpleTimes", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepController>();
  const auto check = [](const auto& st) noexcept {
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

    CHECK(
        st.choose_step(slab.start(), std::numeric_limits<double>::infinity()) ==
        slab.duration());
  };
  check(StepControllers::SimpleTimes{});
  check(*serialize_and_deserialize(
      test_factory_creation<StepController>("  SimpleTimes")));
}
