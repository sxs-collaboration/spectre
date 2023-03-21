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

struct SingleVariableSystem {
  using variables_tag = Var;
};

struct TwoVariableSystem {
  using variables_tag = tmpl::list<Var, AlternativeVar>;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStepId, Var, ::Tags::dt<Var>,
                        Tags::HistoryEvolvedVariables<Var>, AlternativeVar,
                        ::Tags::dt<AlternativeVar>,
                        Tags::HistoryEvolvedVariables<AlternativeVar>>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::RecordTimeStepperData<
                                 typename Metavariables::system_for_test>>>>;
};

template <typename System>
struct Metavariables {
  using system_for_test = System;
  using component_list = tmpl::list<Component<Metavariables>>;
};

template <typename System, bool AlternativeUpdates>
void run_test() {
  using history_tag = Tags::HistoryEvolvedVariables<Var>;
  using alternative_history_tag = Tags::HistoryEvolvedVariables<AlternativeVar>;

  const Slab slab(1., 3.);
  const TimeStepId slab_start_id(true, 0, slab.start());
  const TimeStepId slab_end_id(true, 0, slab.end());

  typename history_tag::type history{};
  history.insert(slab_start_id, -3., 3.);
  typename alternative_history_tag::type alternative_history{};
  alternative_history.insert(slab_start_id, -3., 3.);

  using metavariables = Metavariables<System>;
  using component = Component<metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};

  const double initial_value = 4.;
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {slab_end_id, initial_value, 5., std::move(history), initial_value, 5.,
       std::move(alternative_history)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.template next_action<component>(0);
  auto& box = ActionTesting::get_databox<component>(runner, 0);

  const auto check_history = [&initial_value, &slab_end_id,
                              &slab_start_id](const auto& updated_history) {
    CHECK(updated_history.size() == 2);
    CHECK(updated_history[0].time_step_id == slab_start_id);
    CHECK(updated_history[0].value == std::optional{-3.});
    CHECK(updated_history[0].derivative == 3.);
    CHECK(updated_history[1].time_step_id == slab_end_id);
    CHECK(updated_history[1].value == std::optional{initial_value});
    CHECK(updated_history[1].derivative == 5.);
  };
  check_history(db::get<history_tag>(box));
  if (AlternativeUpdates) {
    check_history(db::get<alternative_history_tag>(box));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.RecordTimeStepperData",
                  "[Unit][Time][Actions]") {
  run_test<SingleVariableSystem, false>();
  run_test<TwoVariableSystem, true>();
}
