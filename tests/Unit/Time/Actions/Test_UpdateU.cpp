// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"               // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
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

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<TimeStepper>>;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStep, variables_tag, history_tag>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::UpdateU<>>>>;
};

template <typename Metavariables>
struct ComponentWithTemplateSpecifiedVariables {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStep, alternative_variables_tag,
                        alternative_history_tag>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::UpdateU<AlternativeVar>>>>;
};

struct Metavariables {
  using system = System;
  using time_stepper_tag = Tags::TimeStepper<TimeStepper>;
  using component_list =
      tmpl::list<Component<Metavariables>,
                 ComponentWithTemplateSpecifiedVariables<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.UpdateU", "[Unit][Time][Actions]") {
  const Slab slab(1., 3.);
  const TimeDelta time_step = slab.duration() / 2;

  const auto rhs = [](const auto t, const auto y) {
    return 2. * t - 2. * (y - t * t);
  };

  using component = Component<Metavariables>;
  using component_with_template_specified_variables =
      ComponentWithTemplateSpecifiedVariables<Metavariables>;
  using simple_tags = typename component::simple_tags;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{std::make_unique<TimeSteppers::RungeKutta3>()}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {time_step, 1., history_tag::type{}});

  ActionTesting::emplace_component_and_initialize<
      component_with_template_specified_variables>(
      &runner, 0, {time_step, 1., alternative_history_tag::type{}});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  const std::array<Time, 3> substep_times{
    {slab.start(), slab.start() + time_step, slab.start() + time_step / 2}};
  // The exact answer is y = x^2, but the integrator would need a
  // smaller step size to get that accurately.
  const std::array<double, 3> expected_values{{3., 3., 10./3.}};

  for (size_t substep = 0; substep < 3; ++substep) {
    auto& before_box = ActionTesting::get_databox<component, simple_tags>(
        make_not_null(&runner), 0);
    db::mutate<history_tag>(
        make_not_null(&before_box),
        [&rhs, &substep, &substep_times ](
            const gsl::not_null<typename history_tag::type*> history,
            const double vars) noexcept {
          const Time& time = gsl::at(substep_times, substep);
          history->insert(TimeStepId(true, 0, time), vars,
                          rhs(time.value(), vars));
        },
        db::get<variables_tag>(before_box));

    auto& alternative_before_box = ActionTesting::get_databox<
        component_with_template_specified_variables,
        typename component_with_template_specified_variables::simple_tags>(
        make_not_null(&runner), 0);
    db::mutate<alternative_history_tag>(
        make_not_null(&alternative_before_box),
        [&rhs, &substep, &substep_times ](
            const gsl::not_null<typename alternative_history_tag::type*>
                alternative_history,
            const double alternative_vars) noexcept {
          const Time& time = gsl::at(substep_times, substep);
          alternative_history->insert(TimeStepId(true, 0, time),
                                      alternative_vars,
                                      rhs(time.value(), alternative_vars));
        },
        db::get<alternative_variables_tag>(alternative_before_box));

    runner.next_action<component>(0);
    runner.next_action<component_with_template_specified_variables>(0);
    const auto& box =
        ActionTesting::get_databox<component, simple_tags>(runner, 0);
    auto& alternative_box = ActionTesting::get_databox<
        component_with_template_specified_variables,
        typename component_with_template_specified_variables::simple_tags>(
        make_not_null(&runner), 0);

    CHECK(db::get<variables_tag>(box) ==
          approx(gsl::at(expected_values, substep)));

    CHECK(db::get<alternative_variables_tag>(alternative_box) ==
          approx(gsl::at(expected_values, substep)));
  }
}
