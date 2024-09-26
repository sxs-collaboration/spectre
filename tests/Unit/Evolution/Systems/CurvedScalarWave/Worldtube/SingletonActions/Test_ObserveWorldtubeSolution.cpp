// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ObserveWorldtubeSolution.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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
                  ::Tags::Time, ::Tags::TimeStepId,
                  Tags::ObserveCoefficientsTrigger, Tags::Psi0, Tags::dtPsi0,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim,
                                       Frame::Inertial>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                                       Frame::Inertial>,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim,
                                       Frame::Inertial>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                                       Frame::Inertial>,
                  Tags::ExcisionSphere<Dim>, Tags::EvolvedPosition<Dim>,
                  Tags::EvolvedVelocity<Dim>,
                  ::Tags::dt<Tags::EvolvedVelocity<Dim>>>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::ObserveWorldtubeSolution>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<
      MockWorldtubeSingleton<MockMetavariables<Dim>>,
      TestHelpers::observers::MockObserverWriter<MockMetavariables<Dim>>>;
  using const_global_cache_tags = tmpl::list<Tags::ExpansionOrder>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Trigger, tmpl::list<Triggers::TimeCompares>>>;
  };
};

template <typename Generator>
void check_observe_worldtube_solution(
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<std::uniform_real_distribution<>*> dist,
    const size_t expansion_order) {
  static constexpr size_t Dim = 3;
  using worldtube_chare = MockWorldtubeSingleton<MockMetavariables<Dim>>;
  using mock_observer_writer =
      TestHelpers::observers::MockObserverWriter<MockMetavariables<Dim>>;
  std::unique_ptr<Trigger> trigger(new Triggers::TimeCompares(
      Options::Comparator::Comparison::GreaterThan, 2.));
  const auto psi0 = make_with_random_values<Scalar<DataVector>>(generator, dist,
                                                                DataVector(1));
  const auto dt_psi0 = make_with_random_values<Scalar<DataVector>>(
      generator, dist, DataVector(1));
  const auto psi_monopole =
      make_with_random_values<Scalar<double>>(generator, dist, 1);
  const auto dt_psi_monopole =
      make_with_random_values<Scalar<double>>(generator, dist, 1);
  const auto psi_dipole =
      make_with_random_values<tnsr::i<double, Dim, Frame::Inertial>>(generator,
                                                                     dist, 1);
  const auto dt_psi_dipole =
      make_with_random_values<tnsr::i<double, Dim, Frame::Inertial>>(generator,
                                                                     dist, 1);

  const auto position =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          generator, dist, DataVector(1));
  const auto velocity =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          generator, dist, DataVector(1));
  const auto acceleration =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          generator, dist, DataVector(1));
  const double wt_radius = 1.5;
  const ExcisionSphere<Dim> excision_sphere(
      wt_radius, tnsr::I<double, Dim, Frame::Grid>{{0., 0., 0.}}, {});

  ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>> runner{
      {expansion_order}};

  // this should NOT trigger because the initial time is too low and the substep
  // is not 0
  const double initial_time_value = 0.5;
  const Slab initial_slab(0.2, 1.2);
  const TimeDelta test_delta(initial_slab, {2, 4});
  const Time initial_time(initial_slab, {2, 4});
  const TimeStepId id(true, 1234, initial_time, 98, test_delta, 1.1);
  ActionTesting::emplace_component_and_initialize<worldtube_chare>(
      make_not_null(&runner), 0,
      {initial_time_value, id, std::move(trigger), psi0, dt_psi0, psi_monopole,
       dt_psi_monopole, psi_dipole, dt_psi_dipole, excision_sphere, position,
       velocity, acceleration});
  ActionTesting::emplace_nodegroup_component_and_initialize<
      mock_observer_writer>(make_not_null(&runner), {});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0);
  // should not trigger, so no action in queue
  CHECK(ActionTesting::is_threaded_action_queue_empty<mock_observer_writer>(
      runner, 0));

  auto& box =
      ActionTesting::get_databox<worldtube_chare>(make_not_null(&runner), 0);
  // mutate the time. This still should NOT trigger because we are still not at
  // the beginning of a slab
  const double new_time_value = 3.9;
  db::mutate<::Tags::Time>(
      [&new_time_value](const gsl::not_null<double*> time) {
        *time = new_time_value;
      },
      make_not_null(&box));
  ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0);
  // should not trigger, so no action in queue
  CHECK(ActionTesting::is_threaded_action_queue_empty<mock_observer_writer>(
      runner, 0));

  // create new time step id with substep 0
  const Slab new_slab(new_time_value, new_time_value + 1.);
  const Time new_time(new_slab, {0, 4});
  const TimeStepId new_id(true, 12345, new_time);
  db::mutate<::Tags::TimeStepId>(
      [&new_id](const gsl::not_null<TimeStepId*> time_step_id) {
        *time_step_id = new_id;
      },
      make_not_null(&box));

  // now we should trigger the action
  ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0);
  CHECK_FALSE(
      ActionTesting::is_threaded_action_queue_empty<mock_observer_writer>(
          runner, 0));
  ActionTesting::invoke_queued_threaded_action<mock_observer_writer>(
      make_not_null(&runner), 0);
  const auto& mock_h5_file = ActionTesting::get_databox_tag<
      mock_observer_writer, TestHelpers::observers::MockReductionFileTag>(
      runner, 0);
  const auto& dat_file = mock_h5_file.get_dat("/PsiTaylorCoefs");
  const auto& data = dat_file.get_data();

  CHECK(data.at(0, 0) == new_time_value);
  CHECK(data.at(0, 1) == get<0>(position)[0]);
  CHECK(data.at(0, 2) == get<1>(position)[0]);
  CHECK(data.at(0, 3) == get<2>(position)[0]);
  CHECK(data.at(0, 4) == get<0>(velocity)[0]);
  CHECK(data.at(0, 5) == get<1>(velocity)[0]);
  CHECK(data.at(0, 6) == get<2>(velocity)[0]);
  CHECK(data.at(0, 7) == get<0>(acceleration)[0]);
  CHECK(data.at(0, 8) == get<1>(acceleration)[0]);
  CHECK(data.at(0, 9) == get<2>(acceleration)[0]);

  if (expansion_order == 0) {
    const std::vector<std::string> legend_0{
        {"Time", "Position_x", "Position_y", "Position_z", "Velocity_x",
         "Velocity_y", "Velocity_z", "Acceleration_x", "Acceleration_y",
         "Acceleration_z", "Psi0", "dtPsi0"}};
    CHECK(legend_0 == dat_file.get_legend());
    CHECK(data.rows() == 1);
    CHECK(data.columns() == 12);
    CHECK(data.at(0, 10) == get(psi_monopole));
    CHECK(data.at(0, 11) == get(dt_psi_monopole));
  } else if (expansion_order == 1) {
    const std::vector<std::string> legend_1{
        {"Time", "Position_x", "Position_y", "Position_z", "Velocity_x",
         "Velocity_y", "Velocity_z", "Acceleration_x", "Acceleration_y",
         "Acceleration_z", "Psi0", "Psix", "Psiy", "Psiz", "dtPsi0", "dtPsix",
         "dtPsiy", "dtPsiz"}};
    CHECK(legend_1 == dat_file.get_legend());
    CHECK(data.rows() == 1);
    CHECK(data.columns() == 18);
    CHECK(data.at(0, 10) == get(psi_monopole));
    CHECK(data.at(0, 11) == get<0>(psi_dipole));
    CHECK(data.at(0, 12) == get<1>(psi_dipole));
    CHECK(data.at(0, 13) == get<2>(psi_dipole));
    CHECK(data.at(0, 14) == get(dt_psi_monopole));
    CHECK(data.at(0, 15) == get<0>(dt_psi_dipole));
    CHECK(data.at(0, 16) == get<1>(dt_psi_dipole));
    CHECK(data.at(0, 17) == get<2>(dt_psi_dipole));
  }
}

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.ObserveWorldtubeSolution",
                  "[Unit]") {
  static constexpr size_t Dim = 3;
  register_factory_classes_with_charm<MockMetavariables<Dim>>();
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);
  check_observe_worldtube_solution(make_not_null(&generator),
                                   make_not_null(&dist), 0);
  check_observe_worldtube_solution(make_not_null(&generator),
                                   make_not_null(&dist), 1);
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
