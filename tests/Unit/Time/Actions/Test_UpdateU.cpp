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
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Actions/UpdateU.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
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

struct System {
  using variables_tag = Var;
};

template <typename Metavariables, typename VariablesTag, typename HistoryTag,
          typename UpdateAction>
struct Component {
  using test_variables_tag = VariablesTag;
  using test_history_tag = HistoryTag;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<Tags::TimeStepper<TimeStepper>, Tags::TimeStep,
                                 ::Tags::IsUsingTimeSteppingErrorControl,
                                 VariablesTag, HistoryTag>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
                 Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<UpdateAction>>>;
};

struct Metavariables;

using component_with_default_variables =
    Component<Metavariables, Var, Tags::HistoryEvolvedVariables<Var>,
              Actions::UpdateU<>>;
using component_with_template_specified_variables =
    Component<Metavariables, AlternativeVar,
              Tags::HistoryEvolvedVariables<AlternativeVar>,
              Actions::UpdateU<AlternativeVar>>;

struct Metavariables {
  using system = System;
  using component_list =
      tmpl::list<component_with_default_variables,
                 component_with_template_specified_variables>;

  void pup(PUP::er& /*p*/) {}
};

template <typename VariablesTag, typename UpdateUArgument>
void test_integration() {
  using history_tag = Tags::HistoryEvolvedVariables<VariablesTag>;

  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  const auto rhs = [](const auto t, const auto y) {
    return 2. * t - 2. * (y - t * t);
  };

  auto box = db::create<db::AddSimpleTags<
      Tags::TimeStepper<TimeSteppers::Rk3HesthavenSsp>, Tags::TimeStep,
      ::Tags::IsUsingTimeSteppingErrorControl, VariablesTag, history_tag>>(
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>(), time_step, false, 1.,
      typename history_tag::type{3});

  const std::array<Time, 3> substep_times{
      {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10. / 3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    db::mutate<history_tag>(
        make_not_null(&box),
        [&rhs, &substep, &substep_times](
            const gsl::not_null<typename history_tag::type*> history,
            const double vars) {
          const double time = gsl::at(substep_times, substep).value();
          history->insert(TimeStepId(true, 0, substep_times[0], substep, time),
                          vars, rhs(time, vars));
        },
        db::get<VariablesTag>(box));

    update_u<System, UpdateUArgument>(make_not_null(&box));
    CHECK(db::get<VariablesTag>(box) ==
          approx(gsl::at(expected_values, substep)));
  }
}

template <typename Component>
void test_action() {
  using variables_tag = typename Component::test_variables_tag;
  using history_tag = typename Component::test_history_tag;

  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  typename variables_tag::type vars = 1.0;
  typename history_tag::type history{3};
  history.insert(TimeStepId(true, 0, slab.start()), vars, 3.0);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<Component>(
      &runner, 0,
      {std::make_unique<TimeSteppers::Rk3HesthavenSsp>(), time_step, false,
       vars, std::move(history)});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto box_for_function = serialize_and_deserialize(
      ActionTesting::get_databox<Component>(make_not_null(&runner), 0));

  runner.next_action<Component>(0);
  update_u<System, variables_tag>(make_not_null(&box_for_function));

  const auto& action_box = ActionTesting::get_databox<Component>(runner, 0);
  CHECK(db::get<variables_tag>(box_for_function) ==
        db::get<variables_tag>(action_box));
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

  const auto do_substep = [&box, &substep_offsets](const Time& step_start,
                                                   const size_t substep) {
    db::mutate<history_tag>(
        make_not_null(&box),
        [&step_start, &substep, &substep_offsets](
            const gsl::not_null<typename history_tag::type*> history,
            const double vars) {
          const Time time = step_start + gsl::at(substep_offsets, substep);
          history->insert(
              TimeStepId(true, 0, step_start, substep, time.value()), vars,
              vars);
        },
        db::get<variables_tag>(box));

    update_u<System>(make_not_null(&box));
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
  Parallel::register_classes_with_charm<TimeSteppers::Rk3HesthavenSsp>();

  test_integration<Var, NoSuchType>();
  test_integration<AlternativeVar, AlternativeVar>();
  test_action<component_with_default_variables>();
  test_action<component_with_template_specified_variables>();
  test_stepper_error();
}
