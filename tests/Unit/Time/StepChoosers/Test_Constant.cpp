// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;
};

using StepChooserType =
    StepChooser<tmpl::list<StepChoosers::Registrars::Constant>>;
using Constant = StepChoosers::Constant<>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Constant", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepChooserType>();

  const Parallel::ConstGlobalCache<Metavariables> cache{{}};
  const auto box = db::create<db::AddSimpleTags<>>();

  const Constant constant{5.4};
  const std::unique_ptr<StepChooserType> constant_base =
      std::make_unique<Constant>(constant);

  CHECK(constant(cache) == 5.4);
  CHECK(constant_base->desired_step(box, cache) == 5.4);
  CHECK(serialize_and_deserialize(constant)(cache) == 5.4);
  CHECK(serialize_and_deserialize(constant_base)->desired_step(box, cache) ==
        5.4);

  test_factory_creation<StepChooserType>("  Constant: 5.4");
}

// [[OutputRegex, Requested step magnitude should be positive]]
SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Constant.bad_create",
                  "[Unit][Time]") {
  ERROR_TEST();
  test_factory_creation<StepChooserType>("  Constant: -5.4");
}
