// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Frac = Time::rational_t;

struct Tag : db::SimpleTag {
  using type = double;
};
using history_tag = Tags::HistoryEvolvedVariables<Tag>;

struct Tag2 : db::SimpleTag {
  using type = double;
};
using history_tag2 = Tags::HistoryEvolvedVariables<Tag2>;

template <typename VariablesTag>
struct System {
  using variables_tag = VariablesTag;
};

using Prevent1 = StepChoosers::PreventRapidIncrease<System<Tag>>;
using Prevent12 =
    StepChoosers::PreventRapidIncrease<System<tmpl::list<Tag, Tag2>>>;
using Prevent21 =
    StepChoosers::PreventRapidIncrease<System<tmpl::list<Tag2, Tag>>>;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<Prevent1, Prevent12, Prevent21>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<Prevent1, Prevent12, Prevent21>>>;
  };
  using component_list = tmpl::list<>;
};

void check_case(const Frac& expected_frac, const std::vector<Frac>& times) {
  CAPTURE(times);
  CAPTURE(expected_frac);

  const Slab slab(0.25, 1.5);
  for (const auto& direction : {1, -1}) {
    CAPTURE(direction);

    const std::optional<double> expected_size =
        expected_frac == -1
            ? std::nullopt
            : std::optional(
                  (direction * expected_frac * slab.duration()).value());

    const auto make_time_id = [&direction, &slab, &times](const size_t i,
                                                          const Frac& mult) {
      Frac frac = -direction * times[i] * mult;
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

    const auto make_gts_time_id = [&direction, &make_time_id](const size_t i) {
      const double time = make_time_id(i, 1).substep_time();
      const double next_time =
          i > 0 ? make_time_id(i - 1, 1).substep_time() : time + direction;
      const Slab gts_slab(std::min(time, next_time), std::max(time, next_time));
      return TimeStepId(direction > 0, -static_cast<int64_t>(i),
                        direction > 0 ? gts_slab.start() : gts_slab.end());
    };

    typename history_tag::type lts_history{};
    typename history_tag::type gts_history{};
    typename history_tag2::type lts_history2{};
    typename history_tag2::type gts_history2{};

    for (size_t i = 1; i < times.size(); ++i) {
      lts_history.insert_initial(make_time_id(i, 1), 0.0, 0.0);
      lts_history2.insert_initial(make_time_id(i, 2), 0.0, 0.0);
      // Setting different step sizes in GTS doesn't make sense.  We
      // still test the logic by using an empty history for Tag2 in
      // one of the tests below.
      gts_history.insert_initial(make_gts_time_id(i), 0.0, 0.0);
      gts_history2.insert_initial(make_gts_time_id(i), 0.0, 0.0);
    }

    const auto check = [&direction, &expected_size](auto use, const auto& box,
                                                    const Time& current_time) {
      using Use = tmpl::type_from<decltype(use)>;
      const auto& history = db::get<history_tag>(box);
      const auto& history2 = db::get<history_tag2>(box);
      const double current_step =
          history.size() > 0
              ? (current_time - history.back().time_step_id.step_time()).value()
              : direction * std::numeric_limits<double>::infinity();

      const Prevent1 relax1{};
      const std::unique_ptr<StepChooser<Use>> relax1_base =
          std::make_unique<Prevent1>(relax1);
      const Prevent12 relax12{};
      const std::unique_ptr<StepChooser<Use>> relax12_base =
          std::make_unique<Prevent12>(relax12);
      const Prevent21 relax21{};
      const std::unique_ptr<StepChooser<Use>> relax21_base =
          std::make_unique<Prevent21>(relax21);

      const TimeStepRequest expected{.size = expected_size};
      CHECK(relax1(history, current_step) == expected);
      CHECK(relax12(history, history2, current_step) == expected);
      CHECK(relax21(history2, history, current_step) == expected);
      CHECK(serialize_and_deserialize(relax1)(history, current_step) ==
            expected);
      CHECK(serialize_and_deserialize(relax12)(history, history2,
                                               current_step) == expected);
      CHECK(serialize_and_deserialize(relax21)(history2, history,
                                               current_step) == expected);
      CHECK(relax1_base->desired_step(current_step, box) == expected);
      CHECK(relax12_base->desired_step(current_step, box) == expected);
      CHECK(relax21_base->desired_step(current_step, box) == expected);
      CHECK(serialize_and_deserialize(relax1_base)
                ->desired_step(current_step, box) == expected);
      CHECK(serialize_and_deserialize(relax12_base)
                ->desired_step(current_step, box) == expected);
      CHECK(serialize_and_deserialize(relax21_base)
                ->desired_step(current_step, box) == expected);
    };

    {
      CAPTURE(lts_history);
      const auto box = db::create<
          db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                            history_tag, history_tag2>>(
          Metavariables{}, lts_history, typename history_tag2::type{});
      check(tmpl::type_<StepChooserUse::LtsStep>{}, box,
            make_time_id(0, 1).step_time());
      check(tmpl::type_<StepChooserUse::Slab>{}, box,
            make_time_id(0, 1).step_time());
    }

    {
      CAPTURE(lts_history);
      CAPTURE(lts_history2);
      const auto box = db::create<
          db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                            history_tag, history_tag2>>(
          Metavariables{}, lts_history, lts_history2);
      check(tmpl::type_<StepChooserUse::LtsStep>{}, box,
            make_time_id(0, 1).step_time());
      check(tmpl::type_<StepChooserUse::Slab>{}, box,
            make_time_id(0, 1).step_time());
    }

    {
      CAPTURE(gts_history);
      const auto box = db::create<
          db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                            history_tag, history_tag2>>(
          Metavariables{}, gts_history, typename history_tag2::type{});
      check(tmpl::type_<StepChooserUse::LtsStep>{}, box,
            make_gts_time_id(0).step_time());
      check(tmpl::type_<StepChooserUse::Slab>{}, box,
            make_gts_time_id(0).step_time());
    }

    {
      CAPTURE(gts_history);
      CAPTURE(gts_history2);
      const auto box = db::create<
          db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                            history_tag, history_tag2>>(
          Metavariables{}, gts_history, gts_history2);
      check(tmpl::type_<StepChooserUse::LtsStep>{}, box,
            make_gts_time_id(0).step_time());
      check(tmpl::type_<StepChooserUse::Slab>{}, box,
            make_gts_time_id(0).step_time());
    }
  }
}

void check_substep_methods() {
  const Slab slab(0.25, 1.5);

  typename history_tag::type history{};

  history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.start(), 1, slab.duration(),
                            (slab.start() + slab.duration() / 3).value()),
                 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.start(), 2, slab.duration(),
                            (slab.start() + slab.duration() / 2).value()),
                 0.0, 0.0);
  const Prevent1 relax{};
  CHECK(relax(history, 3.14) == TimeStepRequest{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.PreventRapidIncrease",
                  "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  // -1 indicates no expected restriction
  check_case(-1, {0});
  check_case(-1, {{3, 8}});
  check_case(-1, {0, {3, 8}});
  check_case(-1, {{4, 8}, {5, 8}, {6, 8}});
  check_case(-1, {{7, 8}, {8, 8}, {9, 8}});
  check_case(-1, {{7, 8}, {8, 8}, {10, 8}});
  check_case({1, 8}, {{7, 8}, {8, 8}, {17, 16}});
  check_case({2, 8}, {{6, 8}, {8, 8}, {9, 8}});
  check_case(-1, {{4, 8}, {5, 8}, {6, 8}, {7, 8}});
  check_case({2, 8}, {{2, 8}, {4, 8}, {5, 8}, {6, 8}});
  check_case({1, 8}, {{4, 8}, {5, 8}, {7, 8}, {8, 8}});

  // Cause roundoff errors
  check_case(-1, {{1, 6}, {2, 6}, {3, 6}});

  check_substep_methods();

  TestHelpers::test_factory_creation<StepChooser<StepChooserUse::LtsStep>,
                                     Prevent1>("PreventRapidIncrease");
  TestHelpers::test_factory_creation<StepChooser<StepChooserUse::Slab>,
                                     Prevent12>("PreventRapidIncrease");

  CHECK(not Prevent1{}.uses_local_data());
  CHECK(not Prevent12{}.uses_local_data());
  CHECK(not Prevent21{}.uses_local_data());
}
