// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Actions/TakeLtsStep.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct System {
  using variables_tag = Var;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<Tags::ConcreteTimeStepper<LtsTimeStepper>,
                 Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                 Tags::StepChoosers, Tags::VariableOrderAlgorithm>;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>,
                        Tags::TimeStep, Tags::AdaptiveSteppingDiagnostics, Var,
                        ::Tags::dt<Var>, Tags::HistoryEvolvedVariables<Var>>;
  using compute_tags = time_stepper_ref_tags<LtsTimeStepper>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::TakeLtsStep<System>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, tmpl::list<>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Time.Actions.TakeLtsStep", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsMoultonPc<false>>();

  std::unique_ptr<LtsTimeStepper> time_stepper =
      std::make_unique<TimeSteppers::AdamsMoultonPc<false>>(2);
  const double minimum_step = 1.0e-8;
  const Slab slab(1., 3.);
  const TimeStepId time_step_id(true, 0, slab.start());
  const auto time_step = slab.duration() / 8;
  const auto next_time_step_id =
      time_stepper->next_time_id(time_step_id, time_step);

  const TimeSteppers::History<double> history{2};

  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{serialize_and_deserialize(time_stepper),
                            EventsAndTriggers{}, Tags::StepChoosers::type{},
                            VariableOrderAlgorithm{2_st}, minimum_step}};

  const double initial_value = 4.0;
  const double derivative = -7.0;

  auto function_box =
      [&](const auto&... box_data) {
        ActionTesting::emplace_component_and_initialize<component>(
            &runner, 0, {box_data...});
        return db::create<
            tmpl::push_back<
                component::simple_tags,
                Parallel::Tags::MetavariablesImpl<Metavariables>,
                Tags::ConcreteTimeStepper<LtsTimeStepper>, Tags::StepChoosers,
                Tags::MinimumTimeStep, Tags::StepperErrorEstimatesEnabled,
                Tags::StepperErrorTolerances<Var>, Tags::StepperErrors<Var>>,
            component::compute_tags>(
            box_data..., Metavariables{}, std::move(time_stepper),
            Tags::StepChoosers::type{}, minimum_step, false,
            Tags::StepperErrorTolerances<Var>::type{},
            Tags::StepperErrors<Var>::type{});
      }(time_step_id, next_time_step_id, time_step,
        Tags::AdaptiveSteppingDiagnostics::type{}, initial_value, derivative,
        history);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  const auto& action_box = ActionTesting::get_databox<component>(runner, 0);

  // Action should be the same as the take_step function.
  take_step<System, true>(make_not_null(&function_box));

  tmpl::for_each<component::simple_tags>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        CHECK(db::get<Tag>(function_box) == db::get<Tag>(action_box));
      });
}
}  // namespace
