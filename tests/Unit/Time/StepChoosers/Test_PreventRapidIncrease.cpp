// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"
// IWYU pragma: no_include "Utilities/Rational.hpp"

namespace {
using Frac = Time::rational_t;
using StepChooserType =
    StepChooser<tmpl::list<StepChoosers::Registrars::PreventRapidIncrease>>;
using PreventRapidIncrease = StepChoosers::PreventRapidIncrease<>;

struct Metavariables {
  using component_list = tmpl::list<>;
};

void check_case(const Frac& expected_frac,
                const std::vector<Frac>& times) noexcept {
  CAPTURE(times);
  CAPTURE(expected_frac);
  const PreventRapidIncrease relax{};
  const std::unique_ptr<StepChooserType> relax_base =
      std::make_unique<PreventRapidIncrease>(relax);

  const Parallel::ConstGlobalCache<Metavariables> cache{{}};

  const Slab slab(0.25, 1.5);
  const double expected = expected_frac == -1
                              ? std::numeric_limits<double>::infinity()
                              : (expected_frac * slab.duration()).value();

  const auto make_time = [&slab](Frac frac) noexcept {
    Slab time_slab = slab;
    while (frac > 1) {
      time_slab = time_slab.advance();
      frac -= 1;
    }
    while (frac < 0) {
      time_slab = time_slab.retreat();
      frac += 1;
    }
    return Time(time_slab, frac);
  };

  for (const auto& sign : {1, -1}) {
    CAPTURE(sign);
    const Time current_time = make_time(sign * times[0]);

    // Silly type.  The step chooser should not care.
    struct Tag : db::SimpleTag {
      using type = std::nullptr_t;
    };
    using history_tag = Tags::HistoryEvolvedVariables<Tag, Tag>;
    db::item_type<history_tag> history{};
    for (size_t i = 1; i < times.size(); ++i) {
      history.insert_initial(make_time(sign * times[i]), nullptr, nullptr);
    }

    CAPTURE(history);
    const auto box =
        db::create<db::AddSimpleTags<Tags::SubstepTime, history_tag>>(
            current_time, std::move(history));

    CHECK(relax(current_time, db::get<history_tag>(box), cache) == expected);
    CHECK(relax_base->desired_step(box, cache) == expected);
    CHECK(serialize_and_deserialize(relax)(
              current_time, db::get<history_tag>(box), cache) == expected);
    CHECK(serialize_and_deserialize(relax_base)->desired_step(box, cache) ==
          expected);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.PreventRapidIncrease",
                  "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepChooserType>();

  // -1 indicates no expected restriction
  check_case(-1, {0});
  check_case(-1, {{2, 5}});
  check_case(-1, {0, {2, 5}});
  check_case(-1, {{1, 5}, {2, 5}, {3, 5}});
  check_case(-1, {{4, 5}, {5, 5}, {6, 5}});
  check_case({1, 5}, {{4, 5}, {5, 5}, {7, 5}});
  check_case({2, 5}, {{3, 5}, {5, 5}, {6, 5}});
  check_case(-1, {{1, 5}, {2, 5}, {3, 5}, {4, 5}});
  check_case({2, 5}, {{0, 5}, {2, 5}, {3, 5}, {4, 5}});
  check_case({1, 20}, {{1, 5}, {1, 4}, {3, 5}, {4, 5}});

  TestHelpers::test_factory_creation<StepChooserType>("PreventRapidIncrease");
}
