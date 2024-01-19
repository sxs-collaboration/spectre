// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Imex/Actions/DoImplicitStep.hpp"
#include "Evolution/Imex/Mode.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Evolution/Imex/Tags/Mode.hpp"
#include "Evolution/Imex/Tags/SolveFailures.hpp"
#include "Evolution/Imex/Tags/SolveTolerance.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/Evolution/Imex/DoImplicitStepSector.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      tmpl::push_front<typename Metavariables::system_tags,
                       Tags::ConcreteTimeStepper<ImexTimeStepper>,
                       Tags::TimeStep, ::Tags::Time,
                       Tags::Next<Tags::TimeStepId>,
                       typename Metavariables::system::variables_tag,
                       imex::Tags::Mode, imex::Tags::SolveTolerance>;

  using compute_tags = time_stepper_ref_tags<ImexTimeStepper>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<imex::Actions::DoImplicitStep<
                                 typename Metavariables::system>>>>;
};

template <typename System, typename SystemTags>
struct Metavariables {
  using system = System;
  using component_list = tmpl::list<Component<Metavariables>>;
  using system_tags = SystemTags;
};

namespace helpers = do_implicit_step_helpers;

void test_basic_functionality() {
  using metavariables = Metavariables<
      helpers::System,
      tmpl::list<imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var1>>,
                 imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var2>>,
                 imex::Tags::SolveFailures<helpers::Sector<helpers::Var1>>,
                 imex::Tags::SolveFailures<helpers::Sector<helpers::Var2>>>>;
  using component = Component<metavariables>;

  const size_t number_of_grid_points = 5;

  const Slab slab(1.0, 3.0);
  const TimeStepId time_step_id(true, 0, slab.start());
  const auto time_step = slab.duration();

  helpers::System::variables_tag::type initial_vars(number_of_grid_points);
  get(get<helpers::Var1>(initial_vars)) = 2.0;
  get(get<helpers::Var2>(initial_vars)) = 3.0;

  imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var1>>::type history1(2);
  history1.insert(time_step_id, decltype(history1)::no_value,
                  -get(get<helpers::Var1>(initial_vars)));
  imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var2>>::type history2(2);
  history2.insert(time_step_id, decltype(history2)::no_value,
                  -get(get<helpers::Var2>(initial_vars)));
  Scalar<DataVector> solve_failures1(DataVector(number_of_grid_points, 0.0));
  Scalar<DataVector> solve_failures2(DataVector(number_of_grid_points, 0.0));

  const double tolerance = 1.0e-10;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {std::make_unique<TimeSteppers::Heun2>(), time_step,
       time_step_id.substep_time(),
       TimeSteppers::Heun2{}.next_time_id(time_step_id, time_step),
       initial_vars, imex::Mode::Implicit, tolerance, std::move(history1),
       std::move(history2), std::move(solve_failures1),
       std::move(solve_failures2)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.next_action<component>(0);

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  CHECK(db::get<::Tags::Time>(box) == time_step_id.substep_time());
  const auto& final_vars = db::get<helpers::System::variables_tag>(box);

  const double dt = time_step.value();
  const double step_factor = (1.0 - 0.5 * dt) / (1.0 + 0.5 * dt);

  CHECK(get(get<helpers::Var1>(final_vars)) ==
        step_factor * get(get<helpers::Var1>(initial_vars)));
  CHECK(get(get<helpers::Var2>(final_vars)) ==
        step_factor * get(get<helpers::Var2>(initial_vars)));
}

void test_nonautonomous() {
  using sector = helpers::NonautonomousSector;
  using Vars = helpers::NonautonomousSystem::variables_tag::type;
  using history_tag = imex::Tags::ImplicitHistory<sector>;
  using metavariables =
      Metavariables<helpers::NonautonomousSystem,
                    tmpl::list<history_tag, imex::Tags::SolveFailures<sector>>>;
  using component = Component<metavariables>;

  const size_t number_of_grid_points = 5;

  const Slab slab(1.0, 3.0);
  const TimeStepId time_step_id(true, 0, slab.start());
  const auto time_step = slab.duration();

  Vars initial_vars(number_of_grid_points);
  get(get<helpers::Var1>(initial_vars)) = 2.0;

  auto expected_var = get<helpers::Var1>(initial_vars);
  // Function is dy/dt = t
  get(expected_var) +=
      0.5 * (square(slab.end().value()) - square(slab.start().value()));

  history_tag::type history(2);
  history.insert(time_step_id, decltype(history)::no_value,
                 make_with_value<history_tag::type::DerivVars>(
                     initial_vars, time_step_id.substep_time()));
  Scalar<DataVector> solve_failures(DataVector(number_of_grid_points, 0.0));

  const double tolerance = 1.0e-10;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {std::make_unique<TimeSteppers::Heun2>(), time_step,
       time_step_id.substep_time(),
       TimeSteppers::Heun2{}.next_time_id(time_step_id, time_step),
       initial_vars, imex::Mode::Implicit, tolerance, std::move(history),
       std::move(solve_failures)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.next_action<component>(0);

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  CHECK(db::get<::Tags::Time>(box) == time_step_id.substep_time());
  const auto& final_var = db::get<helpers::Var1>(box);
  CHECK_ITERABLE_APPROX(final_var, expected_var);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Actions.DoImplicitStep",
                  "[Unit][Evolution][Actions]") {
  register_classes_with_charm<TimeSteppers::Heun2>();
  test_basic_functionality();
  test_nonautonomous();
}
