// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"   // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Time/History.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct AlternativeVar : db::SimpleTag {
  using type = double;
};

struct System {
  using variables_tag = Var;
};

using variables_tag = Var;
using dt_variables_tag = Tags::dt<Var>;
using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;

using alternative_variables_tag = AlternativeVar;
using dt_alternative_variables_tag = Tags::dt<AlternativeVar>;
using alternative_history_tag =
    Tags::HistoryEvolvedVariables<alternative_variables_tag>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags = db::AddSimpleTags<
      Tags::TimeStepId, variables_tag, dt_variables_tag, history_tag>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::RecordTimeStepperData<>>>>;
};

template <typename Metavariables>
struct ComponentWithTemplateSpecifiedVariables {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags = db::AddSimpleTags<
      Tags::TimeStepId, alternative_variables_tag, dt_alternative_variables_tag,
      alternative_history_tag>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::RecordTimeStepperData<AlternativeVar>>>>;
};

struct Metavariables {
  using system = System;
  using component_list =
      tmpl::list<Component<Metavariables>,
                 ComponentWithTemplateSpecifiedVariables<Metavariables>>;
};

template <template <typename> typename LocalComponent, typename VariablesTag,
          typename HistoryTag>
void run_test() {
  const Slab slab(1., 3.);
  const TimeStepId slab_start_id(true, 0, slab.start());
  const TimeStepId slab_end_id(true, 0, slab.end());

  typename HistoryTag::type history{};
  history.insert(slab_start_id, -3., 3.);

  using component = LocalComponent<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};

  const double initial_value = 4.;
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {slab_end_id, initial_value, 5., std::move(history)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.next_action<component>(0);
  auto& box = ActionTesting::get_databox<component>(runner, 0);

  const auto& new_history = db::get<HistoryTag>(box);
  CHECK(new_history.size() == 2);
  CHECK(new_history[0].time_step_id == slab_start_id);
  CHECK(new_history[0].value == std::optional{-3.});
  CHECK(new_history[0].derivative == 3.);
  CHECK(new_history[1].time_step_id == slab_end_id);
  CHECK(new_history[1].value == std::optional{initial_value});
  CHECK(new_history[1].derivative == 5.);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.RecordTimeStepperData",
                  "[Unit][Time][Actions]") {
  run_test<Component, variables_tag, history_tag>();
  run_test<ComponentWithTemplateSpecifiedVariables, alternative_variables_tag,
           alternative_history_tag>();
}
