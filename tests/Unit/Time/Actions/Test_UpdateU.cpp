// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"               // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/IsUsingTimeSteppingErrorControl.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Time/History.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

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
  using simple_tags =
      tmpl::list<Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
                 Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 ::Tags::IsUsingTimeSteppingErrorControl, Var,
                 Tags::HistoryEvolvedVariables<Var>, AlternativeVar,
                 Tags::HistoryEvolvedVariables<AlternativeVar>>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::UpdateU<
                                 typename metavariables::system_for_test>>>>;
};

template <typename System>
struct Metavariables {
  using system_for_test = System;
  using component_list = tmpl::list<Component<Metavariables>>;

  void pup(PUP::er& /*p*/) {}
};

template <typename System, bool AlternativeUpdates>
void test_integration() {
  using history_tag = Tags::HistoryEvolvedVariables<Var>;
  using alternative_history_tag = Tags::HistoryEvolvedVariables<AlternativeVar>;

  const Slab slab(1., 3.);
  const TimeStepId initial_id(true, 0, slab.start());
  const TimeDelta time_step = slab.duration() / 2;
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>();

  const auto rhs = [](const auto t, const auto y) {
    return 2. * t - 2. * (y - t * t);
  };

  auto box =
      db::create<db::AddSimpleTags<
                     Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
                     Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                     ::Tags::IsUsingTimeSteppingErrorControl, Var, history_tag,
                     AlternativeVar, alternative_history_tag>,
                 time_stepper_ref_tags<TimeStepper>>(
          std::move(time_stepper), initial_id,
          time_stepper->next_time_id(initial_id, time_step), time_step, false,
          1., typename history_tag::type{3}, 1.,
          typename alternative_history_tag::type{3});

  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10. / 3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    db::mutate<history_tag, alternative_history_tag>(
        [&rhs](const gsl::not_null<typename history_tag::type*> history,
               const gsl::not_null<typename alternative_history_tag::type*>
                   alternative_history,
               const TimeStepId& time_step_id, const double vars) {
          history->insert(time_step_id, vars,
                          rhs(time_step_id.substep_time(), vars));
          *alternative_history = *history;
        },
        make_not_null(&box), db::get<Tags::TimeStepId>(box), db::get<Var>(box));

    update_u<System>(make_not_null(&box));
    CHECK(db::get<Var>(box) == approx(gsl::at(expected_values, substep)));
    if (AlternativeUpdates) {
      CHECK(db::get<AlternativeVar>(box) ==
            approx(gsl::at(expected_values, substep)));
    } else {
      CHECK(db::get<AlternativeVar>(box) == 1.0);
    }

    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, history_tag>(
        [&time_step](const gsl::not_null<TimeStepId*> time_step_id,
                     const gsl::not_null<TimeStepId*> next_time_step_id,
                     const gsl::not_null<typename history_tag::type*> history,
                     const TimeStepper& stepper) {
          *time_step_id = *next_time_step_id;
          *next_time_step_id = stepper.next_time_id(*time_step_id, time_step);
          stepper.clean_history(history);
        },
        make_not_null(&box), db::get<Tags::TimeStepper<TimeStepper>>(box));
  }
}

void test_action() {
  using system = SingleVariableSystem;
  using metavariables = Metavariables<system>;
  using component = Component<metavariables>;

  const Slab slab(1., 3.);
  const TimeStepId initial_id(true, 0, slab.start());
  const TimeDelta time_step = slab.duration() / 2;
  auto time_stepper = std::make_unique<TimeSteppers::Rk3HesthavenSsp>();

  Var::type vars = 1.0;
  Tags::HistoryEvolvedVariables<Var>::type history{3};
  history.insert(initial_id, vars, 3.0);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {std::move(time_stepper), initial_id,
       time_stepper->next_time_id(initial_id, time_step), time_step, false,
       vars, std::move(history), AlternativeVar::type{},
       Tags::HistoryEvolvedVariables<AlternativeVar>::type{}});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto box_for_function = serialize_and_deserialize(
      ActionTesting::get_databox<component>(make_not_null(&runner), 0));

  runner.next_action<component>(0);
  update_u<system>(make_not_null(&box_for_function));

  const auto& action_box = ActionTesting::get_databox<component>(runner, 0);
  CHECK(db::get<Var>(box_for_function) == db::get<Var>(action_box));
}

void test_stepper_error() {
  using variables_tag = Var;
  using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;

  const Slab slab(1., 3.);
  const TimeStepId initial_id(true, 0, slab.start());
  const TimeDelta initial_time_step = slab.duration() / 2;
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>();

  auto box =
      db::create<db::AddSimpleTags<
                     Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
                     Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                     ::Tags::IsUsingTimeSteppingErrorControl, variables_tag,
                     history_tag, Tags::StepperErrors<variables_tag>>,
                 time_stepper_ref_tags<TimeStepper>>(
          std::move(time_stepper), initial_id,
          time_stepper->next_time_id(initial_id, initial_time_step),
          initial_time_step, true, 1., history_tag::type{3},
          Tags::StepperErrors<variables_tag>::type{});

  const auto do_substep = [&box](const bool repeat_substep = false) {
    db::mutate<history_tag>(
        [](const gsl::not_null<typename history_tag::type*> history,
           const TimeStepId& time_step_id,
           const double vars) { history->insert(time_step_id, vars, vars); },
        make_not_null(&box), db::get<Tags::TimeStepId>(box),
        db::get<variables_tag>(box));

    if (repeat_substep) {
      update_u<SingleVariableSystem>(make_not_null(&box));
    }

    update_u<SingleVariableSystem>(make_not_null(&box));

    db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               history_tag>(
        [](const gsl::not_null<TimeStepId*> time_step_id,
           const gsl::not_null<TimeStepId*> next_time_step_id,
           const gsl::not_null<TimeDelta*> time_step,
           const gsl::not_null<typename history_tag::type*> history,
           const TimeStepper& stepper) {
          *time_step_id = *next_time_step_id;
          *time_step = time_step->with_slab(time_step_id->step_time().slab());
          *next_time_step_id = stepper.next_time_id(*time_step_id, *time_step);
          stepper.clean_history(history);
        },
        make_not_null(&box), db::get<Tags::TimeStepper<TimeStepper>>(box));
  };

  using error_tag = Tags::StepperErrors<variables_tag>;
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  CHECK(not db::get<error_tag>(box)[1].has_value());
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  CHECK(not db::get<error_tag>(box)[1].has_value());
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[1]->step_time == slab.start());

  const auto first_step_error = db::get<error_tag>(box)[1]->error;
  const auto second_step = slab.start() + initial_time_step;
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[1]->step_time == slab.start());
  do_substep();
  CHECK(not db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[1]->step_time == slab.start());
  do_substep(true);
  REQUIRE(db::get<error_tag>(box)[0].has_value());
  REQUIRE(db::get<error_tag>(box)[1].has_value());
  CHECK(db::get<error_tag>(box)[0]->step_time == slab.start());
  CHECK(db::get<error_tag>(box)[1]->step_time == second_step);
  CHECK(db::get<error_tag>(box)[0]->error == first_step_error);
  CHECK(db::get<error_tag>(box)[1]->error != first_step_error);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.UpdateU", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::Rk3HesthavenSsp>();

  test_integration<SingleVariableSystem, false>();
  test_integration<TwoVariableSystem, true>();
  test_action();
  test_stepper_error();
}
