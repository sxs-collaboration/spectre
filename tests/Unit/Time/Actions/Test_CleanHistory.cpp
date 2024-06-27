// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
  using simple_tags = tmpl::conditional_t<
      Metavariables::lts,
      tmpl::list<Tags::ConcreteTimeStepper<LtsTimeStepper>,
                 Tags::HistoryEvolvedVariables<Var>,
                 evolution::dg::Tags::MortarDataHistory<2, double>>,
      tmpl::list<Tags::ConcreteTimeStepper<TimeStepper>,
                 Tags::HistoryEvolvedVariables<Var>,
                 Tags::HistoryEvolvedVariables<AlternativeVar>>>;
  using compute_tags = time_stepper_ref_tags<
      tmpl::conditional_t<Metavariables::lts, LtsTimeStepper, TimeStepper>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::CleanHistory<
              typename metavariables::system_for_test, Metavariables::lts>>>>;
};

template <typename System, bool Lts>
struct Metavariables {
  using system_for_test = System;
  static constexpr bool lts = Lts;
  using component_list = tmpl::list<Component<Metavariables>>;

  void pup(PUP::er& /*p*/) {}
};

template <bool TwoVars>
void test_action() {
  using system =
      tmpl::conditional_t<TwoVars, TwoVariableSystem, SingleVariableSystem>;
  using metavariables = Metavariables<system, false>;
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

void test_lts() {
  using metavariables = Metavariables<SingleVariableSystem, true>;
  using component = Component<metavariables>;

  const Slab slab(1., 3.);
  TimeSteppers::History<double> history{2};
  history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.end()), 0.0, 0.0);

  TimeSteppers::BoundaryHistory<evolution::dg::MortarData<2>,
                                evolution::dg::MortarData<2>, double>
      boundary_history{};
  boundary_history.local().insert(TimeStepId(true, 0, slab.start()), 2, {});
  boundary_history.local().insert(TimeStepId(true, 0, slab.end()), 2, {});
  boundary_history.remote().insert(TimeStepId(true, 0, slab.start()), 2, {});
  boundary_history.remote().insert(TimeStepId(true, 0, slab.end()), 2, {});
  evolution::dg::Tags::MortarDataHistory<2, double>::type mortar_histories{};
  const std::array mortars{
      DirectionalId<2>{{Direction<2>::Axis::Xi, Side::Lower}, ElementId<2>{}},
      DirectionalId<2>{{Direction<2>::Axis::Xi, Side::Upper}, ElementId<2>{}}};
  for (const auto& mortar : mortars) {
    mortar_histories.emplace(mortar, boundary_history);
  }

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {std::make_unique<TimeSteppers::AdamsBashforth>(2), std::move(history),
       std::move(mortar_histories)});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  CHECK(db::get<Tags::HistoryEvolvedVariables<Var>>(box).size() == 1);
  for (const auto& mortar : mortars) {
    CHECK(db::get<evolution::dg::Tags::MortarDataHistory<2, double>>(box)
              .at(mortar)
              .local()
              .size() == 1);
  }
}

struct FieldVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct ImexSystem : tt::ConformsTo<imex::protocols::ImexSystem> {
  using variables_tag = Tags::Variables<tmpl::list<FieldVar>>;

  struct Sector : tt::ConformsTo<imex::protocols::ImplicitSector> {
    using tensors = tmpl::list<FieldVar>;
    using initial_guess = imex::GuessExplicitResult;
    struct Attempt {
      using tags_from_evolution = tmpl::list<>;
      using simple_tags = tmpl::list<>;
      using compute_tags = tmpl::list<>;
      struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                      tt::ConformsTo<::protocols::StaticReturnApplyable> {
        using return_tags = tmpl::list<Tags::Source<FieldVar>>;
        using argument_tags = tmpl::list<>;
        static void apply(gsl::not_null<Scalar<DataVector>*> var_source);
      };
      using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;
      using source_prep = tmpl::list<>;
      using jacobian_prep = tmpl::list<>;
    };
    using solve_attempts = tmpl::list<Attempt>;
  };

  using implicit_sectors = tmpl::list<Sector>;
};

static_assert(
    tt::assert_conforms_to_v<ImexSystem, imex::protocols::ImexSystem>);

struct ImexMetavariables;

struct ImexComponent {
  using metavariables = ImexMetavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags =
      tmpl::list<Tags::ConcreteTimeStepper<ImexTimeStepper>,
                 Tags::HistoryEvolvedVariables<ImexSystem::variables_tag>,
                 imex::Tags::ImplicitHistory<ImexSystem::Sector>>;
  using compute_tags = time_stepper_ref_tags<ImexTimeStepper>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::CleanHistory<ImexSystem, false>>>>;
};

struct ImexMetavariables {
  using component_list = tmpl::list<ImexComponent>;

  void pup(PUP::er& /*p*/) {}
};

void test_imex() {
  const Slab slab(1., 3.);
  TimeSteppers::History<Variables<tmpl::list<FieldVar>>> history{2};
  history.insert(TimeStepId(true, 0, slab.start()), {5, 0.0}, {5, 0.0});
  history.insert(
      TimeStepId(true, 0, slab.start(), 1, slab.duration(), slab.end().value()),
      {5, 0.0}, {5, 0.0});
  history.insert(TimeStepId(true, 1, slab.end()), {5, 0.0}, {5, 0.0});

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<ImexMetavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<ImexComponent>(
      &runner, 0, {std::make_unique<TimeSteppers::Heun2>(), history, history});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  ActionTesting::next_action<ImexComponent>(make_not_null(&runner), 0);

  const auto& box = ActionTesting::get_databox<ImexComponent>(runner, 0);
  CHECK(db::get<Tags::HistoryEvolvedVariables<ImexSystem::variables_tag>>(box)
            .size() == 1);
  CHECK(db::get<imex::Tags::ImplicitHistory<ImexSystem::Sector>>(box).size() ==
        1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.CleanHistory", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth,
                              TimeSteppers::Heun2>();

  test_action<false>();
  test_action<true>();
  test_lts();
  test_imex();
}
