// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
// IWYU pragma: no_include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = double;
};

struct System {
  using variables_tag = Var;
};

struct Metavariables;
using component =
    ActionTesting::MockArrayComponent<Metavariables, int, tmpl::list<>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.RecordTimeStepperData",
                  "[Unit][Time][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};
  using variables_tag = Var;
  using dt_variables_tag = Tags::dt<Var>;

  const Slab slab(1., 3.);
  const TimeId time_id(true, 8, slab.start());

  using history_tag =
      Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>;
  history_tag::type history{};
  history.insert(slab.end(), 2., 3.);

  auto box = db::create<db::AddSimpleTags<Tags::TimeId, variables_tag,
                                          dt_variables_tag, history_tag>,
                        db::AddComputeTags<Tags::Time>>(time_id, 4., 5.,
                                                        std::move(history));

  box = std::get<0>(
      runner.apply<component, Actions::RecordTimeStepperData>(box, 0));

  const auto& new_history = db::get<history_tag>(box);
  CHECK(new_history.size() == 2);
  CHECK(*new_history.begin() == slab.end());
  CHECK(new_history.begin().value() == 2.);
  CHECK(new_history.begin().derivative() == 3.);
  CHECK(*(new_history.begin() + 1) == slab.start());
  CHECK((new_history.begin() + 1).value() == 4.);
  CHECK((new_history.begin() + 1).derivative() == 5.);
}
