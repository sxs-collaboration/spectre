// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Components/KleinGordonCharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/KleinGordonBoundaryTestHelpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"

namespace Cce {

namespace {

template <typename Metavariables>
struct mock_observer_writer {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using simple_tags = tmpl::list<observers::Tags::H5FileLock>;

  using const_global_cache_tags = tmpl::list<>;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct mock_klein_gordon_h5_worldtube_boundary {
  using component_being_mocked = KleinGordonH5WorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list =
      tmpl::list<Actions::InitializeWorldtubeBoundary<
          KleinGordonH5WorldtubeBoundary<Metavariables>>>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

template <typename Metavariables>
struct mock_klein_gordon_characteristic_evolution {
  using component_being_mocked =
      KleinGordonCharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list = tmpl::list<
      Actions::InitializeKleinGordonVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionTime<
          typename Metavariables::evolved_coordinates_variables_tag,
          typename Metavariables::evolved_swsh_tags, false>,
      // advance the time so that the current `TimeStepId` is valid without
      // having to perform self-start.
      ::Actions::AdvanceTime, Parallel::Actions::TerminatePhase>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<
              Actions::RequestBoundaryData<
                  KleinGordonH5WorldtubeBoundary<Metavariables>,
                  mock_klein_gordon_characteristic_evolution<Metavariables>>,
              Actions::ReceiveWorldtubeData<
                  Metavariables,
                  typename Metavariables::cce_boundary_communication_tags>,
              Actions::ReceiveWorldtubeData<
                  Metavariables, typename Metavariables::
                                     klein_gordon_boundary_communication_tags>,
              Actions::RequestNextBoundaryData<
                  KleinGordonH5WorldtubeBoundary<Metavariables>,
                  mock_klein_gordon_characteristic_evolution<Metavariables>>>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct test_metavariables : CharacteristicExtractDefaults<false> {
  using cce_base = CharacteristicExtractDefaults<false>;
  using evolved_swsh_tags = tmpl::append<cce_base::evolved_swsh_tags,
                                         tmpl::list<Cce::Tags::KleinGordonPsi>>;
  using evolved_swsh_dt_tags =
      tmpl::append<cce_base::evolved_swsh_dt_tags,
                   tmpl::list<Cce::Tags::KleinGordonPi>>;
  using cce_step_choosers = tmpl::list<>;
  using scri_values_to_observe = tmpl::list<>;

  using klein_gordon_boundary_communication_tags =
      Cce::Tags::klein_gordon_worldtube_boundary_tags;
  using klein_gordon_gauge_boundary_tags = tmpl::list<
      Cce::Tags::EvolutionGaugeBoundaryValue<Cce::Tags::KleinGordonPsi>,
      Cce::Tags::EvolutionGaugeBoundaryValue<Cce::Tags::KleinGordonPi>>;
  using klein_gordon_scri_tags =
      tmpl::list<Cce::Tags::ScriPlus<Cce::Tags::KleinGordonPsi>,
                 Cce::Tags::ScriPlus<Cce::Tags::KleinGordonPi>>;

  using klein_gordon_pre_swsh_derivative_tags = tmpl::list<>;
  using klein_gordon_swsh_derivative_tags = tmpl::list<>;
  using klein_gordon_transform_buffer_tags = tmpl::list<>;
  using klein_gordon_source_tags = tmpl::list<>;
  using klein_gordon_cce_integrand_tags = tmpl::list<>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, tmpl::list<>>>;
  };

  using component_list =
      tmpl::list<mock_klein_gordon_h5_worldtube_boundary<test_metavariables>,
                 mock_klein_gordon_characteristic_evolution<test_metavariables>,
                 mock_observer_writer<test_metavariables>>;
};

// This function tests the communication between the evolution and boundary
// components of `KleinGordonCharacteristicExtract`, which passes around
// worldtube data for Klein-Gordon CCE evolution.
//
// The core check flow is as follows:
//  (a) The evolution component asks the boundary component to send worldtube
//  data once the data are available (`RequestBoundaryData`)
//  (b) When the data are not ready, the evolution is paused and waits for data.
//  Here we check that the evolution component is indeed terminated.
//  (c) Then the boundary component prepares and sends the data.
//  (d) The evolution component receives the data.
//  (e) We check that the received data are the same as the ones we generated at
//  the beginning.
template <typename Generator>
void test_klein_gordon_h5_boundary_communication(
    const gsl::not_null<Generator*> gen) {
  using evolution_component =
      mock_klein_gordon_characteristic_evolution<test_metavariables>;
  using worldtube_component =
      mock_klein_gordon_h5_worldtube_boundary<test_metavariables>;
  using writer_component = mock_observer_writer<test_metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;

  const std::string filename =
      "KleinGordonH5BoundaryCommunicationTestCceR0100.h5";

  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  // create the test file, because on initialization the manager will need to
  // get basic data out of the file
  TestHelpers::write_scalar_tensor_test_file(solution, filename, target_time,
                                             extraction_radius, frequency,
                                             amplitude, l_max);

  const double start_time = target_time;
  const double target_step_size = 0.01 * value_dist(*gen);
  const double end_time = std::numeric_limits<double>::quiet_NaN();

  // tests start here
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {l_max, extraction_radius,
       Tags::EndTimeFromFile::create_from_options(end_time, filename, false),
       start_time, number_of_radial_points}};

  const size_t buffer_size = 5;
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<writer_component>(
      &runner, 0, {Parallel::NodeLock{}});
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size,
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<::TimeSteppers::AdamsBashforth>(3)),
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(),
      target_step_size);
  ActionTesting::emplace_component<worldtube_component>(
      &runner, 0,
      Tags::H5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          false, false, std::optional<double>{}),
      Tags::KleinGordonH5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          std::optional<double>{}));

  // this should run the initializations
  for (size_t i = 0; i < 6; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<worldtube_component>(make_not_null(&runner), 0);
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);

  // ask the boundary component for data
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  // the first request from the evolution component (for tensor data)
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  // now the evolution component should be paused since the data are not ready
  CHECK(ActionTesting::get_terminate<evolution_component>(runner, 0));

  // the response of the boundary component
  // (`BoundaryComputeAndSendToEvolution`): prepare boundary data for tensor and
  // scalar variables
  ActionTesting::invoke_queued_simple_action<worldtube_component>(
      make_not_null(&runner), 0);

  // rerun the two `ReceiveWorldtubeData` to receive tensor and scalar data
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);

  // finally, run `RequestNextBoundaryData`. Since `end_time` is NaN (see
  // above), `BoundaryComputeAndSendToEvolution` is not called.
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);

  // Then we check that the received data are consistent with what we generated
  //
  // first, the tensor variables
  const size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};
  TestHelpers::create_fake_time_varying_modal_data(
      make_not_null(&spatial_metric_coefficients),
      make_not_null(&dt_spatial_metric_coefficients),
      make_not_null(&dr_spatial_metric_coefficients),
      make_not_null(&shift_coefficients), make_not_null(&dt_shift_coefficients),
      make_not_null(&dr_shift_coefficients), make_not_null(&lapse_coefficients),
      make_not_null(&dt_lapse_coefficients),
      make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
      amplitude, frequency, target_time, l_max, false);

  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      expected_boundary_variables{number_of_angular_points};
  create_bondi_boundary_data(
      make_not_null(&expected_boundary_variables), spatial_metric_coefficients,
      dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
      shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
      lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
      extraction_radius, l_max);

  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&expected_boundary_variables, &runner,
       &angular_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs =
            ActionTesting::get_databox_tag<evolution_component, tag>(runner, 0);
        const auto& test_rhs = get<tag>(expected_boundary_variables);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs,
                                     angular_derivative_approx);
      });

  // then the scalar variables
  Scalar<ComplexModalVector> expected_kg_psi_modal;
  Scalar<ComplexModalVector> expected_kg_pi_modal;
  Scalar<DataVector> expected_kg_psi_nodal;
  Scalar<DataVector> expected_kg_pi_nodal;

  TestHelpers::create_fake_time_varying_klein_gordon_data(
      make_not_null(&expected_kg_psi_modal),
      make_not_null(&expected_kg_pi_modal),
      make_not_null(&expected_kg_psi_nodal),
      make_not_null(&expected_kg_pi_nodal), extraction_radius, amplitude,
      frequency, target_time, l_max);

  const auto& kg_psi_from_actions =
      ActionTesting::get_databox_tag<evolution_component,
                                     Tags::BoundaryValue<Tags::KleinGordonPsi>>(
          runner, 0);

  const auto& kg_pi_from_actions =
      ActionTesting::get_databox_tag<evolution_component,
                                     Tags::BoundaryValue<Tags::KleinGordonPi>>(
          runner, 0);

  // convert `expected_kg_psi_nodal` and `expected_kg_pi_nodal` to
  // `ComplexDataVector`.
  ComplexDataVector expected_kg_psi_nodal_complex{get(expected_kg_psi_nodal)};
  ComplexDataVector expected_kg_pi_nodal_complex{get(expected_kg_pi_nodal)};

  CHECK_ITERABLE_CUSTOM_APPROX(get(kg_psi_from_actions).data(),
                               expected_kg_psi_nodal_complex,
                               angular_derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(kg_pi_from_actions).data(),
                               expected_kg_pi_nodal_complex,
                               angular_derivative_approx);
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.KleinGordonH5BoundaryCommunication",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_klein_gordon_h5_boundary_communication(make_not_null(&gen));
}
}  // namespace Cce
