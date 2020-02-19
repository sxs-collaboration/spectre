// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Parallel/PhaseDependentActionList.hpp"   // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

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
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStepId, variables_tag, dt_variables_tag,
                        history_tag>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Actions::RecordTimeStepperData<>>>>;
};

template <typename Metavariables>
struct ComponentWithTemplateSpecifiedVariables {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStepId, alternative_variables_tag,
                        dt_alternative_variables_tag, alternative_history_tag>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Actions::RecordTimeStepperData<AlternativeVar>>>>;
};

struct Metavariables {
  using system = System;
  using component_list =
      tmpl::list<Component<Metavariables>,
                 ComponentWithTemplateSpecifiedVariables<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.RecordTimeStepperData",
                  "[Unit][Time][Actions]") {
  const Slab slab(1., 3.);

  history_tag::type history{};
  history.insert(TimeStepId(true, 0, slab.end()), 2., 3.);

  alternative_history_tag::type alternative_history{};
  alternative_history.insert(TimeStepId(true, 0, slab.end()), 2., 3.);

  using component = Component<Metavariables>;
  using component_with_template_specified_variables =
      ComponentWithTemplateSpecifiedVariables<Metavariables>;
  using simple_tags = typename component::simple_tags;
  using compute_tags = typename component::compute_tags;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeStepId(true, 0, slab.start()), 4., 5., std::move(history)});
  ActionTesting::emplace_component_and_initialize<
      component_with_template_specified_variables>(
      &runner, 0,
      {TimeStepId(true, 0, slab.start()), 4., 5.,
       std::move(alternative_history)});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.next_action<component>(0);
  runner.next_action<component_with_template_specified_variables>(0);
  auto& box =
      ActionTesting::get_databox<component,
                                 tmpl::append<simple_tags, compute_tags>>(
          runner, 0);
  auto& template_specified_variables_box = ActionTesting::get_databox<
      component_with_template_specified_variables,
      tmpl::append<
          typename component_with_template_specified_variables::simple_tags,
          typename component_with_template_specified_variables::compute_tags>>(
      runner, 0);

  const auto& new_history = db::get<history_tag>(box);
  CHECK(new_history.size() == 2);
  CHECK(*new_history.begin() == slab.end());
  CHECK(new_history.begin().value() == 2.);
  CHECK(new_history.begin().derivative() == 3.);
  CHECK(*(new_history.begin() + 1) == slab.start());
  CHECK((new_history.begin() + 1).value() == 4.);
  CHECK((new_history.begin() + 1).derivative() == 5.);

  const auto& new_history_from_template_specified_variables_box =
      db::get<alternative_history_tag>(template_specified_variables_box);
  CHECK(new_history_from_template_specified_variables_box.size() == 2);
  CHECK(*new_history_from_template_specified_variables_box.begin() ==
        slab.end());
  CHECK(new_history_from_template_specified_variables_box.begin().value() ==
        2.);
  CHECK(
      new_history_from_template_specified_variables_box.begin().derivative() ==
      3.);
  CHECK(*(new_history_from_template_specified_variables_box.begin() + 1) ==
        slab.start());
  CHECK(
      (new_history_from_template_specified_variables_box.begin() + 1).value() ==
      4.);
  CHECK((new_history_from_template_specified_variables_box.begin() + 1)
            .derivative() == 5.);
}
