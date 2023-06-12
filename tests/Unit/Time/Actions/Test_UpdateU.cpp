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
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Time/History.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

class TimeStepper;
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
      tmpl::list<Tags::TimeStepper<TimeStepper>, Tags::TimeStep,
                 ::Tags::IsUsingTimeSteppingErrorControl, Var,
                 Tags::HistoryEvolvedVariables<Var>, AlternativeVar,
                 Tags::HistoryEvolvedVariables<AlternativeVar>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
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
  const TimeDelta time_step = slab.duration() / 2;

  const auto rhs = [](const auto t, const auto y) {
    return 2. * t - 2. * (y - t * t);
  };

  auto box = db::create<db::AddSimpleTags<
      Tags::TimeStepper<TimeSteppers::Rk3HesthavenSsp>, Tags::TimeStep,
      ::Tags::IsUsingTimeSteppingErrorControl, Var, history_tag, AlternativeVar,
      alternative_history_tag>>(
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>(), time_step, false, 1.,
      typename history_tag::type{3}, 1.,
      typename alternative_history_tag::type{3});

  const std::array<Time, 3> substep_times{
      {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10. / 3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    db::mutate<history_tag, alternative_history_tag>(
        [&rhs, &substep, &substep_times, &time_step](
            const gsl::not_null<typename history_tag::type*> history,
            const gsl::not_null<typename alternative_history_tag::type*>
                alternative_history,
            const double vars) {
          const double time = gsl::at(substep_times, substep).value();
          history->insert(
              TimeStepId(true, 0, substep_times[0], substep, time_step, time),
              vars, rhs(time, vars));
          *alternative_history = *history;
        },
        make_not_null(&box), db::get<Var>(box));

    update_u<System>(make_not_null(&box));
    CHECK(db::get<Var>(box) == approx(gsl::at(expected_values, substep)));
    if (AlternativeUpdates) {
      CHECK(db::get<AlternativeVar>(box) ==
            approx(gsl::at(expected_values, substep)));
    } else {
      CHECK(db::get<AlternativeVar>(box) == 1.0);
    }
  }
}

void test_action() {
  using system = SingleVariableSystem;
  using metavariables = Metavariables<system>;
  using component = Component<metavariables>;

  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  Var::type vars = 1.0;
  Tags::HistoryEvolvedVariables<Var>::type history{3};
  history.insert(TimeStepId(true, 0, slab.start()), vars, 3.0);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {std::make_unique<TimeSteppers::Rk3HesthavenSsp>(), time_step, false,
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
  const TimeDelta time_step = slab.duration() / 2;

  auto box = db::create<db::AddSimpleTags<
      Tags::TimeStepper<TimeSteppers::Rk3HesthavenSsp>, Tags::TimeStep,
      ::Tags::IsUsingTimeSteppingErrorControl, variables_tag, history_tag,
      Tags::StepperError<variables_tag>,
      Tags::PreviousStepperError<variables_tag>, Tags::StepperErrorUpdated>>(
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>(), time_step, true, 1.,
      history_tag::type{3}, 1234.5, 1234.5, false);

  const std::array<TimeDelta, 3> substep_offsets{
      {0 * slab.duration(), time_step, time_step / 2}};

  const auto do_substep = [&box, &substep_offsets, &time_step](
                              const Time& step_start, const size_t substep) {
    db::mutate<history_tag>(
        [&step_start, &substep, &substep_offsets, &time_step](
            const gsl::not_null<typename history_tag::type*> history,
            const double vars) {
          const Time time = step_start + gsl::at(substep_offsets, substep);
          history->insert(
              TimeStepId(true, 0, step_start, substep, time_step, time.value()),
              vars, vars);
        },
        make_not_null(&box), db::get<variables_tag>(box));

    update_u<SingleVariableSystem>(make_not_null(&box));
  };

  using error_tag = Tags::StepperError<variables_tag>;
  using previous_error_tag = Tags::PreviousStepperError<variables_tag>;
  do_substep(slab.start(), 0);
  CHECK(not db::get<Tags::StepperErrorUpdated>(box));
  do_substep(slab.start(), 1);
  CHECK(not db::get<Tags::StepperErrorUpdated>(box));
  do_substep(slab.start(), 2);
  CHECK(db::get<Tags::StepperErrorUpdated>(box));
  CHECK(db::get<error_tag>(box) != 1234.5);

  const auto first_step_error = db::get<error_tag>(box);
  const auto second_step = slab.start() + time_step;
  do_substep(second_step, 0);
  CHECK(not db::get<Tags::StepperErrorUpdated>(box));
  do_substep(second_step, 1);
  CHECK(not db::get<Tags::StepperErrorUpdated>(box));
  do_substep(second_step, 2);
  CHECK(db::get<Tags::StepperErrorUpdated>(box));
  CHECK(db::get<error_tag>(box) != first_step_error);
  CHECK(db::get<previous_error_tag>(box) == first_step_error);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.UpdateU", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::Rk3HesthavenSsp>();

  test_integration<SingleVariableSystem, false>();
  test_integration<TwoVariableSystem, true>();
  test_action();
  test_stepper_error();
}
