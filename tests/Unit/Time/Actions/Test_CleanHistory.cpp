// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
struct er;
}  // namespace PUP

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
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<Tags::ConcreteTimeStepper<TimeStepper>,
                                 Tags::HistoryEvolvedVariables<Var>,
                                 Tags::HistoryEvolvedVariables<AlternativeVar>>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::CleanHistory<
                                 typename metavariables::system_for_test>>>>;
};

template <typename System>
struct Metavariables {
  using system_for_test = System;
  using component_list = tmpl::list<Component<Metavariables>>;

  void pup(PUP::er& /*p*/) {}
};

template <bool TwoVars>
void test_action() {
  using system =
      tmpl::conditional_t<TwoVars, TwoVariableSystem, SingleVariableSystem>;
  using metavariables = Metavariables<system>;
  using component = Component<metavariables>;

  const Slab slab(1., 3.);
  TimeSteppers::History<double> history{2};
  history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.end()), 0.0, 0.0);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {std::make_unique<TimeSteppers::AdamsBashforth>(2), history, history});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  CHECK(db::get<Tags::HistoryEvolvedVariables<Var>>(box).size() == 1);
  CHECK(db::get<Tags::HistoryEvolvedVariables<AlternativeVar>>(box).size() ==
        (TwoVars ? 1 : 2));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.CleanHistory", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();

  test_action<false>();
  test_action<true>();
}
