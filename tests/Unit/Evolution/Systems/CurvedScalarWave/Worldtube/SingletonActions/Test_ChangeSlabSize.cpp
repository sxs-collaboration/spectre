// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ChangeSlabSize.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

template <typename Metavariables>
struct MockWorldtubeSingleton {
  using metavariables = Metavariables;
  static constexpr size_t Dim = metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<
                  ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                  ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                  ::Tags::TimeStepper<TimeStepper>>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::ChangeSlabSize>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list =
      tmpl::list<MockWorldtubeSingleton<MockMetavariables<Dim>>>;
  using const_global_cache_tags = tmpl::list<>;
};

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.ChangeSlabSize", "[Unit]") {
  static constexpr size_t Dim = 3;
  using worldtube_chare = MockWorldtubeSingleton<MockMetavariables<Dim>>;
  register_classes_with_charm<TimeSteppers::Rk3HesthavenSsp>();

  const auto time_stepper = TimeSteppers::Rk3HesthavenSsp{};
  const double slab_1_start = 1.5;
  const double slab_1_end = 2.0;

  const Slab slab_1(slab_1_start, slab_1_end);
  const auto start_time_1 = slab_1.start();
  const TimeStepId time_step_id_1(true, 1, start_time_1);
  const auto step_1 = time_step_id_1.step_time().slab().duration();

  // we use two element ids for this test
  const auto element_ids =
      initial_element_ids(0, std::array<size_t, Dim>{{1, 0, 0}});

  // we need to create a new runner every time an ASSERT is triggered because
  // they can not be copied.
  const auto create_new_runner =
      [&time_step_id_1, &time_stepper,
       &step_1](gsl::not_null<
                ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>>*>
                    runner) {
        ActionTesting::emplace_component_and_initialize<worldtube_chare>(
            runner, 0,
            {time_step_id_1, time_stepper.next_time_id(time_step_id_1, step_1),
             step_1, step_1,
             std::make_unique<TimeSteppers::Rk3HesthavenSsp>()});
        ActionTesting::set_phase(runner, Parallel::Phase::Testing);
      };
  const auto check_time_tags =
      [](const ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>>& runner,
         const TimeStepId& time_step_id, const TimeStepId& next_time_step_id,
         const TimeDelta& step, const TimeDelta& next_step) {
        CHECK(
            ActionTesting::get_databox_tag<worldtube_chare, ::Tags::TimeStepId>(
                runner, 0) == time_step_id);
        CHECK(ActionTesting::get_databox_tag<worldtube_chare,
                                             ::Tags::Next<::Tags::TimeStepId>>(
                  runner, 0) == next_time_step_id);
        CHECK(ActionTesting::get_databox_tag<worldtube_chare, ::Tags::TimeStep>(
                  runner, 0) == step);
        CHECK(ActionTesting::get_databox_tag<worldtube_chare,
                                             ::Tags::Next<::Tags::TimeStep>>(
                  runner, 0) == next_step);
      };
  using inbox_variables_type =
      Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                           ::Tags::dt<CurvedScalarWave::Tags::Psi>>>;
  {
    ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>> runner{{}};
    create_new_runner(make_not_null(&runner));
    // ChangeSlabSize but no data sent yet
    CHECK_FALSE(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    check_time_tags(runner, time_step_id_1,
                    time_stepper.next_time_id(time_step_id_1, step_1), step_1,
                    step_1);
    auto& worldtube_inbox =
        ActionTesting::get_inbox_tag<worldtube_chare,
                                     Tags::SphericalHarmonicsInbox<Dim>>(
            make_not_null(&runner), 0);
    worldtube_inbox[time_step_id_1][element_ids.at(0)] = inbox_variables_type{};
    // ChangeSlabSize but only 1/2 elements have sent
    CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    check_time_tags(runner, time_step_id_1,
                    time_stepper.next_time_id(time_step_id_1, step_1), step_1,
                    step_1);
    // send from other element
    worldtube_inbox[time_step_id_1][element_ids.at(1)] = inbox_variables_type{};
    CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    // same slab size so nothing shoud have changed.
    check_time_tags(runner, time_step_id_1,
                    time_stepper.next_time_id(time_step_id_1, step_1), step_1,
                    step_1);
#ifdef SPECTRE_DEBUG
    const TimeStepId time_step_id_new_slab_number(true, 2, start_time_1);
    // sending a second time step id, should fail
    worldtube_inbox[time_step_id_new_slab_number][element_ids.at(0)] =
        inbox_variables_type{};
    CHECK_THROWS_WITH(
        ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0),
        Catch::Contains("Received data from two different time step ids."));
#endif
  }
#ifdef SPECTRE_DEBUG
  {
    ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>> runner{{}};
    create_new_runner(make_not_null(&runner));
    auto& worldtube_inbox =
        ActionTesting::get_inbox_tag<worldtube_chare,
                                     Tags::SphericalHarmonicsInbox<Dim>>(
            make_not_null(&runner), 0);
    // different slab start -> invalid
    const double slab_2_start = 1.7;
    const Slab slab_2(slab_2_start, slab_1_end);
    const auto start_time_2 = slab_2.start();
    const TimeStepId time_step_id_2(true, 1, start_time_2);
    worldtube_inbox[time_step_id_2][element_ids.at(0)] = inbox_variables_type{};
    CHECK_THROWS_WITH(
        ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0),
        Catch::Contains(
            "The new slab should start at the same time as the old one."));
  }
#endif

  {
    ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>> runner{{}};
    create_new_runner(make_not_null(&runner));
    auto& worldtube_inbox =
        ActionTesting::get_inbox_tag<worldtube_chare,
                                     Tags::SphericalHarmonicsInbox<Dim>>(
            make_not_null(&runner), 0);
    // different slab end
    const double slab_2_end = 1.7;
    const Slab slab_2(slab_1_start, slab_2_end);
    const auto start_time_2 = slab_2.start();
    const TimeStepId time_step_id_2(true, 1, start_time_2);
    const auto step_2 = time_step_id_2.step_time().slab().duration();
    worldtube_inbox[time_step_id_2][element_ids.at(0)] = inbox_variables_type{};
    // data sent from only one element but with a different time step id, so the
    // time tags should have changed.
    CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    check_time_tags(runner, time_step_id_2,
                    time_stepper.next_time_id(time_step_id_2, step_2), step_2,
                    step_2);
    // data sent from both elements
    worldtube_inbox[time_step_id_2][element_ids.at(1)] = inbox_variables_type{};
    CHECK(ActionTesting::next_action_if_ready<worldtube_chare>(
        make_not_null(&runner), 0));
    check_time_tags(runner, time_step_id_2,
                    time_stepper.next_time_id(time_step_id_2, step_2), step_2,
                    step_2);
  }
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
