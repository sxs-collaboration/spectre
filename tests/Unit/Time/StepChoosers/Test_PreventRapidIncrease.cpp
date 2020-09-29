// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

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

  const Parallel::GlobalCache<Metavariables> cache{{}};

  const Slab slab(0.25, 1.5);
  const double expected = expected_frac == -1
                              ? std::numeric_limits<double>::infinity()
                              : (expected_frac * slab.duration()).value();

  for (const auto& direction : {1, -1}) {
    CAPTURE(direction);

    const auto make_time_id =
        [&direction, &slab, &times](const size_t i) noexcept {
      Frac frac = -direction * times[i];
      int64_t slab_number = 0;
      Slab time_slab = slab;
      while (frac > 1) {
        time_slab = time_slab.advance();
        frac -= 1;
        slab_number += direction;
      }
      while (frac < 0) {
        time_slab = time_slab.retreat();
        frac += 1;
        slab_number -= direction;
      }
      return TimeStepId(direction > 0, slab_number, Time(time_slab, frac));
    };

    // Silly type.  The step chooser should not care.
    struct Tag : db::SimpleTag {
      using type = std::nullptr_t;
    };
    using history_tag = Tags::HistoryEvolvedVariables<Tag>;
    typename history_tag::type lts_history{};
    typename history_tag::type gts_history{};

    const auto make_gts_time_id =
        [&direction, &make_time_id](const size_t i) noexcept {
      const double time = make_time_id(i).substep_time().value();
      const double next_time =
          i > 0 ? make_time_id(i - 1).substep_time().value() : time + direction;
      const Slab gts_slab(std::min(time, next_time), std::max(time, next_time));
      return TimeStepId(direction > 0, -static_cast<int64_t>(i),
                        direction > 0 ? gts_slab.start() : gts_slab.end());
    };

    for (size_t i = 1; i < times.size(); ++i) {
      lts_history.insert_initial(make_time_id(i), nullptr, nullptr);
      gts_history.insert_initial(make_gts_time_id(i), nullptr, nullptr);
    }

    const auto check = [&cache, &expected, &relax, &relax_base](
        const auto& box, const Time& current_time) noexcept {
      const auto& history = db::get<history_tag>(box);
      const double current_step =
          history.size() > 0 ? abs(current_time - history.back()).value()
                             : std::numeric_limits<double>::infinity();

      CHECK(relax(history, current_step, cache) == expected);
      CHECK(relax_base->desired_step(current_step, box, cache) == expected);
      CHECK(serialize_and_deserialize(relax)(history, current_step, cache) ==
            expected);
      CHECK(serialize_and_deserialize(relax_base)
                ->desired_step(current_step, box, cache) == expected);
    };

    {
      CAPTURE(lts_history);
      check(db::create<db::AddSimpleTags<history_tag>>(std::move(lts_history)),
            make_time_id(0).substep_time());
    }

    {
      CAPTURE(gts_history);
      check(db::create<db::AddSimpleTags<history_tag>>(std::move(gts_history)),
            make_gts_time_id(0).substep_time());
    }
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

  // Cause roundoff errors
  check_case(-1, {{1, 3}, {2, 3}, {3, 3}});

  TestHelpers::test_factory_creation<StepChooserType>("PreventRapidIncrease");
}
