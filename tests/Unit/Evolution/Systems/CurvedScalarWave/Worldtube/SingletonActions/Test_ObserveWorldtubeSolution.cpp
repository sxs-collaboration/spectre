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
#include "Domain/Structure/ExcisionSphere.hpp"
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
#include "Time/Tags.hpp"
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
                  ::Tags::Time, Tags::ObserveCoefficientsTrigger, Tags::Psi0,
                  Tags::dtPsi0,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                                       Frame::Grid>,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                                       Frame::Grid>,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 2, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 2, Dim,
                                       Frame::Grid>,
                  Tags::ExcisionSphere<Dim>>,
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
      make_with_random_values<tnsr::i<double, Dim, Frame::Grid>>(generator,
                                                                 dist, 1);
  const auto dt_psi_dipole =
      make_with_random_values<tnsr::i<double, Dim, Frame::Grid>>(generator,
                                                                 dist, 1);
  const auto psi_quadrupole =
      make_with_random_values<tnsr::ii<double, Dim, Frame::Grid>>(generator,
                                                                  dist, 1);
  const auto dt_psi_quadrupole =
      make_with_random_values<tnsr::ii<double, Dim, Frame::Grid>>(generator,
                                                                  dist, 1);
  const double wt_radius = 1.5;
  const ExcisionSphere<Dim> excision_sphere(
      wt_radius, tnsr::I<double, Dim, Frame::Grid>{{0., 0., 0.}}, {});

  ActionTesting::MockRuntimeSystem<MockMetavariables<Dim>> runner{
      {expansion_order}};

  // initial time should not trigger
  const double initial_time = 0.5;
  ActionTesting::emplace_component_and_initialize<worldtube_chare>(
      make_not_null(&runner), 0,
      {initial_time, std::move(trigger), psi0, dt_psi0, psi_monopole,
       dt_psi_monopole, psi_dipole, dt_psi_dipole, psi_quadrupole,
       dt_psi_quadrupole, excision_sphere});
  ActionTesting::emplace_nodegroup_component_and_initialize<
      mock_observer_writer>(make_not_null(&runner), {});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0);
  // should not trigger, so no action in queue
  CHECK(ActionTesting::is_threaded_action_queue_empty<mock_observer_writer>(
      runner, 0));
  auto& box =
      ActionTesting::get_databox<worldtube_chare>(make_not_null(&runner), 0);

  // mutate time so we trigger an observation
  const double new_time = 3.9;
  db::mutate<::Tags::Time>(
      [&new_time](const gsl::not_null<double*> time) { *time = new_time; },
      make_not_null(&box));

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

  if (expansion_order == 0) {
    const std::vector<std::string> legend_0{{"Time", "Psi0", "dtPsi0"}};
    CHECK(legend_0 == dat_file.get_legend());
    const auto& data = dat_file.get_data();
    CHECK(data.rows() == 1);
    CHECK(data.columns() == 3);
    CHECK(data.at(0, 0) == new_time);
    CHECK(data.at(0, 1) == get(psi_monopole));
    CHECK(data.at(0, 2) == get(dt_psi_monopole));
  } else if (expansion_order == 1) {
    const std::vector<std::string> legend_1{{"Time", "Psi0", "Psix", "Psiy",
                                             "Psiz", "dtPsi0", "dtPsix",
                                             "dtPsiy", "dtPsiz"}};
    CHECK(legend_1 == dat_file.get_legend());
    const auto& data = dat_file.get_data();
    CHECK(data.rows() == 1);
    CHECK(data.columns() == 9);
    CHECK(data.at(0, 0) == new_time);
    CHECK(data.at(0, 1) == get(psi_monopole));
    CHECK(data.at(0, 2) == get<0>(psi_dipole));
    CHECK(data.at(0, 3) == get<1>(psi_dipole));
    CHECK(data.at(0, 4) == get<2>(psi_dipole));
    CHECK(data.at(0, 5) == get(dt_psi_monopole));
    CHECK(data.at(0, 6) == get<0>(dt_psi_dipole));
    CHECK(data.at(0, 7) == get<1>(dt_psi_dipole));
    CHECK(data.at(0, 8) == get<2>(dt_psi_dipole));
  } else if (expansion_order == 2) {
    const std::vector<std::string> legend_2{
        {"Time",    "Psi0",    "Psix",   "Psiy",    "Psiz",    "Psixx",
         "Psixy",   "Psixz",   "Psiyy",  "Psiyz",   "Psizz",   "dtPsi0",
         "dtPsix",  "dtPsiy",  "dtPsiz", "dtPsixx", "dtPsixy", "dtPsixz",
         "dtPsiyy", "dtPsiyz", "dtPsizz"}};
    CHECK(legend_2 == dat_file.get_legend());
    const auto& data = dat_file.get_data();
    CHECK(data.rows() == 1);
    CHECK(data.columns() == 21);
    CHECK(data.at(0, 0) == new_time);
    CHECK(data.at(0, 1) == get(psi0).at(0));
    CHECK(data.at(0, 2) == get<0>(psi_dipole));
    CHECK(data.at(0, 3) == get<1>(psi_dipole));
    CHECK(data.at(0, 4) == get<2>(psi_dipole));
    Approx custom_approx = Approx::custom().epsilon(1.e-14).scale(1.0);

    const double trace_psi_2 =
        3. * (get(psi_monopole) - get(psi0).at(0)) / square(wt_radius);
    CHECK(data.at(0, 5) ==
          custom_approx(get<0, 0>(psi_quadrupole) + trace_psi_2));
    CHECK(data.at(0, 6) == get<0, 1>(psi_quadrupole));
    CHECK(data.at(0, 7) == get<0, 2>(psi_quadrupole));
    CHECK(data.at(0, 8) ==
          custom_approx(get<1, 1>(psi_quadrupole) + trace_psi_2));
    CHECK(data.at(0, 9) == get<1, 2>(psi_quadrupole));
    CHECK(data.at(0, 10) ==
          custom_approx(get<2, 2>(psi_quadrupole) + trace_psi_2));

    CHECK(data.at(0, 11) == get(dt_psi0).at(0));
    CHECK(data.at(0, 12) == get<0>(dt_psi_dipole));
    CHECK(data.at(0, 13) == get<1>(dt_psi_dipole));
    CHECK(data.at(0, 14) == get<2>(dt_psi_dipole));

    const double trace_dt_psi_2 =
        3. * (get(dt_psi_monopole) - get(dt_psi0).at(0)) / square(wt_radius);
    CHECK(data.at(0, 15) ==
          custom_approx(get<0, 0>(dt_psi_quadrupole) + trace_dt_psi_2));
    CHECK(data.at(0, 16) == get<0, 1>(dt_psi_quadrupole));
    CHECK(data.at(0, 17) == get<0, 2>(dt_psi_quadrupole));
    CHECK(data.at(0, 18) ==
          custom_approx(get<1, 1>(dt_psi_quadrupole) + trace_dt_psi_2));
    CHECK(data.at(0, 19) == get<1, 2>(dt_psi_quadrupole));
    CHECK(data.at(0, 20) ==
          custom_approx(get<2, 2>(dt_psi_quadrupole) + trace_dt_psi_2));
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
  check_observe_worldtube_solution(make_not_null(&generator),
                                   make_not_null(&dist), 2);
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
