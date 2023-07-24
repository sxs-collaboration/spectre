// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::EventsAndTriggers>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 typename Metavariables::simple_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::RunEventsAndTriggers<
                                 typename Metavariables::observation_id>>>>;
};

template <typename ObservationId, typename SimpleTags>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using observation_id = ObservationId;
  using simple_tags = SimpleTags;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::Completion>>,
                  tmpl::pair<Trigger, Triggers::logical_triggers>>;
  };
};

struct Label {};

using iteration_metavars =
    Metavariables<Convergence::Tags::IterationId<Label>,
                  tmpl::list<Convergence::Tags::IterationId<Label>>>;

using time_metavars =
    Metavariables<Tags::Time, tmpl::list<Tags::Time, Tags::TimeStepId>>;

void check_trigger(const bool expected, std::unique_ptr<Trigger> trigger) {
  EventsAndTriggers::Storage events_and_triggers_input;
  events_and_triggers_input.push_back(
      {std::move(trigger), make_vector<std::unique_ptr<Event>>(
                               std::make_unique<Events::Completion>())});

  using my_component = Component<iteration_metavars>;
  ActionTesting::MockRuntimeSystem<iteration_metavars> runner{
      {EventsAndTriggers(std::move(events_and_triggers_input))}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  db::mutate<Convergence::Tags::IterationId<Label>>(
      [](const gsl::not_null<size_t*> n) { *n = 10; },
      make_not_null(&ActionTesting::get_databox<my_component>(
          make_not_null(&runner), 0)));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);

  CHECK(ActionTesting::get_terminate<my_component>(runner, 0) == expected);
}

void test_slab_limits() {
  EventsAndTriggers::Storage events_and_triggers_input;
  events_and_triggers_input.push_back(
      {std::make_unique<Triggers::Always>(),
       make_vector<std::unique_ptr<Event>>(
           std::make_unique<Events::Completion>())});

  const Slab slab(0.0, 1.0);
  const auto start = slab.start();
  const auto center = start + slab.duration() / 2;

  using my_component = Component<time_metavars>;
  ActionTesting::MockRuntimeSystem<time_metavars> runner{
      {EventsAndTriggers(std::move(events_and_triggers_input))}};
  ActionTesting::emplace_component<my_component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& box =
      ActionTesting::get_databox<my_component>(make_not_null(&runner), 0);
  db::mutate<Tags::Time>([](const gsl::not_null<double*> t) { *t = -10.0; },
                         make_not_null(&box));

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, center);
      },
      make_not_null(&box));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  CHECK(not ActionTesting::get_terminate<my_component>(runner, 0));

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, start, 1, slab.duration(), start.value());
      },
      make_not_null(&box));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  CHECK(not ActionTesting::get_terminate<my_component>(runner, 0));

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(true, 0, start);
      },
      make_not_null(&box));
  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_terminate<my_component>(runner, 0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.RunEventsAndTriggers",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<iteration_metavars>();

  check_trigger(true, std::make_unique<Triggers::Always>());
  check_trigger(false, std::make_unique<Triggers::Not>(
                           std::make_unique<Triggers::Always>()));

  test_slab_limits();
}
