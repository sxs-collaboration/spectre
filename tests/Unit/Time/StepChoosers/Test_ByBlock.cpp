// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "ParallelBackend/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_include "ParallelBackend/PupStlCpp11.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ByBlock", "[Unit][Time]") {
  constexpr size_t volume_dim = 2;

  using StepChooserType =
      StepChooser<tmpl::list<StepChoosers::Registrars::ByBlock<volume_dim>>>;
  using ByBlock = StepChoosers::ByBlock<volume_dim>;

  Parallel::register_derived_classes_with_charm<StepChooserType>();

  const ByBlock by_block({2.5, 3.0, 3.5});
  const std::unique_ptr<StepChooserType> by_block_base =
      std::make_unique<ByBlock>(by_block);

  const Parallel::ConstGlobalCache<Metavariables> cache{{}};
  for (size_t block = 0; block < 3; ++block) {
    const Element<volume_dim> element(ElementId<volume_dim>(block), {});
    const auto box =
        db::create<db::AddSimpleTags<Tags::Element<volume_dim>>>(element);
    const double expected = 0.5 * (block + 5);

    CHECK(by_block(element, cache) == expected);
    CHECK(by_block_base->desired_step(box, cache) == expected);
    CHECK(serialize_and_deserialize(by_block)(element, cache) == expected);
    CHECK(serialize_and_deserialize(by_block_base)->desired_step(box, cache) ==
          expected);
  }

  test_factory_creation<StepChooserType>(
      "  ByBlock:\n"
      "    Sizes: [1.0, 2.0]");
}
