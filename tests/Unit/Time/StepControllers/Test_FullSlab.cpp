// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>

#include "Time/Slab.hpp"
#include "Time/StepControllers/FullSlab.hpp"
#include "Time/Time.hpp"
#include "tests/Unit/TestCreation.hpp"

class StepController;

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

  test_factory_creation<StepController>("  FullSlab");
}
