// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_include "Time/History.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = double;
};

struct System {
  using variables_tag = Var;
};

using variables_tag = Var;
using dt_variables_tag = Tags::dt<Var>;
using history_tag =
    Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags = db::AddSimpleTags<Tags::SubstepTime, variables_tag,
                                        dt_variables_tag, history_tag>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::RecordTimeStepperData>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.RecordTimeStepperData",
                  "[Unit][Time][Actions]") {
  const Slab slab(1., 3.);

  history_tag::type history{};
  history.insert(slab.end(), 2., 3.);

  using component = Component<Metavariables>;
  using simple_tags = typename component::simple_tags;
  using compute_tags = typename component::compute_tags;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {slab.start(), 4., 5., std::move(history)});
  runner.set_phase(Metavariables::Phase::Testing);
  runner.next_action<component>(0);
  auto& box =
      ActionTesting::get_databox<component,
                                 tmpl::append<simple_tags, compute_tags>>(
          runner, 0);

  const auto& new_history = db::get<history_tag>(box);
  CHECK(new_history.size() == 2);
  CHECK(*new_history.begin() == slab.end());
  CHECK(new_history.begin().value() == 2.);
  CHECK(new_history.begin().derivative() == 3.);
  CHECK(*(new_history.begin() + 1) == slab.start());
  CHECK((new_history.begin() + 1).value() == 4.);
  CHECK((new_history.begin() + 1).derivative() == 5.);
}
