// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>
// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {
using step_choosers = tmpl::list<StepChoosers::Registrars::Constant>;
using change_step_size = Actions::ChangeStepSize<step_choosers>;

struct Var : db::SimpleTag {
  using type = double;
};

using history_tag = Tags::HistoryEvolvedVariables<Var>;

struct System {
  using variables_tag = Var;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<LtsTimeStepper>>;
  using simple_tags = tmpl::list<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                                 Tags::TimeStep, Tags::Next<Tags::TimeStep>,
                                 history_tag, typename System::variables_tag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<change_step_size,
                     /*UpdateU action is required to satisfy internal checks of
                        `ChangeStepSize`. It is not used in the test.*/
                     Actions::UpdateU<>>>>;
};

struct Metavariables {
  using system = System;
  static constexpr bool local_time_stepping = true;
  using const_global_cache_tags = change_step_size::const_global_cache_tags;
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

void check(const bool time_runs_forward,
           std::unique_ptr<LtsTimeStepper> time_stepper, const Time& time,
           const double request, const TimeDelta& expected_step) noexcept {
  CAPTURE(time);
  CAPTURE(request);

  const TimeDelta initial_step_size =
      (time_runs_forward ? 1 : -1) * time.slab().duration();

  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{
      {make_vector<std::unique_ptr<StepChooser<step_choosers>>>(
           std::make_unique<StepChoosers::Constant<step_choosers>>(2. *
                                                                   request),
           std::make_unique<StepChoosers::Constant<step_choosers>>(request),
           std::make_unique<StepChoosers::Constant<step_choosers>>(2. *
                                                                   request)),
       std::make_unique<StepControllers::BinaryFraction>(),
       std::move(time_stepper)}};

  // Initialize the component
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeStepId(
           time_runs_forward, -1,
           (time_runs_forward ? time.slab().end() : time.slab().start()) -
               initial_step_size),
       TimeStepId(time_runs_forward, 0, time), initial_step_size,
       initial_step_size, typename history_tag::type{}, 1.});

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.next_action<component>(0);
  const auto& box =
      ActionTesting::get_databox<component, typename component::simple_tags>(
          runner, 0);

  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == expected_step);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeStepSize", "[Unit][Time][Actions]") {
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  Parallel::register_derived_classes_with_charm<StepChooser<step_choosers>>();
  Parallel::register_derived_classes_with_charm<StepController>();
  const Slab slab(-5., -2.);
  const double slab_length = slab.duration().value();
  check(true, std::make_unique<TimeSteppers::AdamsBashforthN>(1),
        slab.start() + slab.duration() / 4, slab_length / 5.,
        slab.duration() / 8);
  check(true, std::make_unique<TimeSteppers::AdamsBashforthN>(1),
        slab.start() + slab.duration() / 4, slab_length, slab.duration() / 4);
  check(false, std::make_unique<TimeSteppers::AdamsBashforthN>(1),
        slab.end() - slab.duration() / 4, slab_length / 5.,
        -slab.duration() / 8);
  check(false, std::make_unique<TimeSteppers::AdamsBashforthN>(1),
        slab.end() - slab.duration() / 4, slab_length, -slab.duration() / 4);
}
