// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <initializer_list>  // IWYU pragma: keep
#include <memory>
// IWYU pragma: no_include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Increase", "[Unit][Time]") {
  using registrars = tmpl::list<StepChoosers::Register::Increase>;
  using Increase = StepChoosers::Increase<registrars>;

  Parallel::register_derived_classes_with_charm<StepChooser<registrars>>();

  const Parallel::ConstGlobalCache<Metavariables> cache{{}};
  for (const auto& sign : {1, -1}) {
    const auto step = sign * Slab(0., 1.).duration() / 4;
    const auto box = db::create<db::AddSimpleTags<Tags::TimeStep>>(step);

    const Increase increase{5.};
    const std::unique_ptr<StepChooser<registrars>> increase_base =
        std::make_unique<Increase>(increase);

    CHECK(increase(step, cache) == 1.25);
    CHECK(increase_base->desired_step(box, cache) == 1.25);
    CHECK(serialize_and_deserialize(increase)(step, cache) == 1.25);
    CHECK(serialize_and_deserialize(increase_base)->desired_step(box, cache) ==
          1.25);
  }

  test_factory_creation<StepChooser<registrars>>(
      "  Increase:\n"
      "    Factor: 5.0");
}
