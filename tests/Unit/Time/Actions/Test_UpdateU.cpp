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
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Time/History.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

class TimeStepper;

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
using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;


using alternative_variables_tag = AlternativeVar;
using dt_alternative_variables_tag = Tags::dt<AlternativeVar>;
using alternative_history_tag =
    Tags::HistoryEvolvedVariables<alternative_variables_tag>;

template <typename Metavariables, typename SimpleTags, typename UpdateAction>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<TimeStepper>>;
  using simple_tags = tmpl::append<
      tmpl::list<Tags::TimeStep, ::Tags::IsUsingTimeSteppingErrorControl>,
      SimpleTags>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
                 Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<UpdateAction>>>;
};

struct Metavariables;

using component_with_default_variables =
    Component<Metavariables, tmpl::list<variables_tag, history_tag>,
              Actions::UpdateU<>>;
using component_with_template_specified_variables =
    Component<Metavariables,
              tmpl::list<alternative_variables_tag, alternative_history_tag>,
              Actions::UpdateU<AlternativeVar>>;
using component_with_stepper_error = Component<
    Metavariables,
    tmpl::list<variables_tag, history_tag, Tags::StepperError<variables_tag>,
               Tags::PreviousStepperError<variables_tag>,
               Tags::StepperErrorUpdated>,
    Actions::UpdateU<>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component_with_default_variables,
                                    component_with_template_specified_variables,
                                    component_with_stepper_error>;
};

void test_integration() {
  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  const auto rhs = [](const auto t, const auto y) {
    return 2. * t - 2. * (y - t * t);
  };

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::Rk3HesthavenSsp>()}};
  ActionTesting::emplace_component_and_initialize<
      component_with_default_variables>(
      &runner, 0, {time_step, false, 1., history_tag::type{3}});

  ActionTesting::emplace_component_and_initialize<
      component_with_template_specified_variables>(
      &runner, 0, {time_step, false, 1., alternative_history_tag::type{3}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const std::array<Time, 3> substep_times{
    {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10./3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    auto& before_box =
        ActionTesting::get_databox<component_with_default_variables>(
            make_not_null(&runner), 0);
    db::mutate<history_tag>(
        make_not_null(&before_box),
        [&rhs, &substep, &substep_times](
            const gsl::not_null<typename history_tag::type*> history,
            const double vars) {
          const Time& time = gsl::at(substep_times, substep);
          history->insert(TimeStepId(true, 0, substep_times[0], substep, time),
                          rhs(time.value(), vars));
        },
        db::get<variables_tag>(before_box));

    auto& alternative_before_box =
        ActionTesting::get_databox<component_with_template_specified_variables>(
            make_not_null(&runner), 0);
    db::mutate<alternative_history_tag>(
        make_not_null(&alternative_before_box),
        [&rhs, &substep, &substep_times](
            const gsl::not_null<typename alternative_history_tag::type*>
                alternative_history,
            const double alternative_vars) {
          const Time& time = gsl::at(substep_times, substep);
          alternative_history->insert(
              TimeStepId(true, 0, substep_times[0], substep, time),
              rhs(time.value(), alternative_vars));
        },
        db::get<alternative_variables_tag>(alternative_before_box));

    runner.next_action<component_with_default_variables>(0);
    runner.next_action<component_with_template_specified_variables>(0);
    const auto& box =
        ActionTesting::get_databox<component_with_default_variables>(runner, 0);
    auto& alternative_box =
        ActionTesting::get_databox<component_with_template_specified_variables>(
            make_not_null(&runner), 0);

    CHECK(db::get<variables_tag>(box) ==
          approx(gsl::at(expected_values, substep)));

    CHECK(db::get<alternative_variables_tag>(alternative_box) ==
          approx(gsl::at(expected_values, substep)));
  }
}

void test_stepper_error() {
  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::Rk3HesthavenSsp>()}};
  ActionTesting::emplace_component_and_initialize<component_with_stepper_error>(
      &runner, 0,
      {time_step, true, 1., history_tag::type{3}, 1234.5, 1234.5, false});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const std::array<TimeDelta, 3> substep_offsets{
      {0 * slab.duration(), time_step, time_step / 2}};

  auto& box = ActionTesting::get_databox<component_with_stepper_error>(
      make_not_null(&runner), 0);
  const auto do_substep = [&box, &runner, &substep_offsets](
                              const Time& step_start, const size_t substep) {
    db::mutate<history_tag>(
        make_not_null(&box),
        [&step_start, &substep, &substep_offsets](
            const gsl::not_null<typename history_tag::type*> history,
            const double vars) {
          const Time time = step_start + gsl::at(substep_offsets, substep);
          history->insert(TimeStepId(true, 0, step_start, substep, time), vars);
        },
        db::get<variables_tag>(box));

    runner.next_action<component_with_stepper_error>(0);
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

  test_integration();
  test_stepper_error();
}
