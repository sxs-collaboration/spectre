// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Time.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
constexpr size_t volume_dim = 2;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::ByBlock<
                                 StepChooserUse::LtsStep, volume_dim>>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<StepChoosers::ByBlock<
                                 StepChooserUse::Slab, volume_dim>>>>;
  };
  using component_list = tmpl::list<>;
};

template <typename Use>
void test_by_block() {
  using ByBlock = StepChoosers::ByBlock<Use, volume_dim>;

  const ByBlock by_block({2.5, 3.0, 3.5});
  const std::unique_ptr<StepChooser<Use>> by_block_base =
      std::make_unique<ByBlock>(by_block);

  const double current_step = std::numeric_limits<double>::infinity();
  const Parallel::GlobalCache<Metavariables> cache{};
  for (size_t block = 0; block < 3; ++block) {
    const Element<volume_dim> element(ElementId<volume_dim>(block), {});
    auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          domain::Tags::Element<volume_dim>>>(Metavariables{},
                                                              element);
    const double expected = 0.5 * static_cast<double>(block + 5);

    CHECK(by_block(element, current_step, cache) ==
          std::make_pair(expected, true));
    CHECK(serialize_and_deserialize(by_block)(element, current_step, cache) ==
          std::make_pair(expected, true));

    if constexpr (std::is_same_v<Use, StepChooserUse::LtsStep>) {
      CHECK(by_block_base->desired_step(make_not_null(&box), current_step,
                                        cache) ==
            std::make_pair(expected, true));
      CHECK(serialize_and_deserialize(by_block_base)
                ->desired_step(make_not_null(&box), current_step, cache) ==
            std::make_pair(expected, true));
    } else {
      CHECK(serialize_and_deserialize(by_block_base)
                ->desired_slab(current_step, box, cache) == expected);
      CHECK(by_block_base->desired_slab(current_step, box, cache) == expected);
    }
  }

  TestHelpers::test_factory_creation<StepChooser<Use>, ByBlock>(
      "ByBlock:\n"
      "  Sizes: [1.0, 2.0]");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ByBlock", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  test_by_block<StepChooserUse::LtsStep>();
  test_by_block<StepChooserUse::Slab>();
}
