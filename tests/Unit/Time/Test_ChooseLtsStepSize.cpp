// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"

SPECTRE_TEST_CASE("Unit.Time.ChooseLtsStepSize", "[Unit][Time]") {
  const Slab slab(1., 4.);
  CHECK(choose_lts_step_size(slab.start(), 4.) == slab.duration());
  CHECK(choose_lts_step_size(slab.start(), 10.) == slab.duration());
  CHECK(choose_lts_step_size(slab.start(), 2.) == slab.duration() / 2);
  CHECK(choose_lts_step_size(slab.start(), 1.4) == slab.duration() / 4);
  CHECK(choose_lts_step_size(slab.start() + slab.duration() / 4, 2.) ==
        slab.duration() / 4);

  CHECK(choose_lts_step_size(slab.end(), -2.) == -slab.duration() / 2);

  CHECK(choose_lts_step_size(slab.start(),
                             std::numeric_limits<double>::infinity()) ==
        slab.duration());

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      choose_lts_step_size(slab.start() + slab.duration() / 3, 1.),
      Catch::Contains("Not at a binary-fraction time within slab"));
#endif  // SPECTRE_DEBUG
}
