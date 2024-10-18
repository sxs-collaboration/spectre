// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Systems/Cce/Components/KleinGordonCharacteristicEvolution.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/Actions/CharacteristicInitialization.hpp"
#include "Helpers/Evolution/Systems/Cce/KleinGordonBoundaryTestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/MakeVector.hpp"

namespace Cce {
namespace {

template <typename Metavariables>
struct mock_klein_gordon_characteristic_evolution {
  using component_being_mocked =
      KleinGordonCharacteristicEvolution<Metavariables>;

  using initialize_action_list = tmpl::list<
      Actions::InitializeKleinGordonVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionTime<
          typename Metavariables::evolved_coordinates_variables_tag,
          typename Metavariables::evolved_swsh_tags, false>,
      // advance the time so that the current `TimeStepId` is valid without
      // having to perform self-start.
      ::Actions::AdvanceTime,
      Actions::InitializeCharacteristicEvolutionScri<
          typename Metavariables::scri_values_to_observe, NoSuchType>,
      Parallel::Actions::TerminatePhase>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct metavariables : CharacteristicExtractDefaults<false> {
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

  using klein_gordon_pre_swsh_derivative_tags =
      tmpl::list<Cce::Tags::Dy<Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>>,
                 Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>>;
  using klein_gordon_swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::EthEth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::EthEthbar>>;
  using klein_gordon_transform_buffer_tags = tmpl::list<
      Spectral::Swsh::Tags::SwshTransform<Cce::Tags::KleinGordonPsi>,
      Spectral::Swsh::Tags::SwshTransform<
          Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::Eth>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::Ethbar>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>, Spectral::Swsh::Tags::Eth>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>,
          Spectral::Swsh::Tags::Ethbar>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::EthEth>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::EthEthbar>>>;

  using klein_gordon_source_tags = tmpl::flatten<
      tmpl::transform<Cce::bondi_hypersurface_step_tags,
                      tmpl::bind<Cce::Tags::KleinGordonSource, tmpl::_1>>>;

  using klein_gordon_cce_integrand_tags =
      tmpl::list<Cce::Tags::PoleOfIntegrand<Cce::Tags::KleinGordonPi>,
                 Cce::Tags::RegularIntegrand<Cce::Tags::KleinGordonPi>>;

  using const_global_cache_tags = tmpl::list<Tags::SpecifiedStartTime>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, tmpl::list<>>>;
  };

  using component_list =
      tmpl::list<mock_klein_gordon_characteristic_evolution<metavariables>>;
};

// This function tests the `initialize_action_list` of the
// `KleinGordonCharacteristicEvolution` component, which initializes the data
// storage for all tags of the component.
//
// The function begins by generating and storing some scalar and tensor data
// into an HDF5 file named `filename` using `write_scalar_tensor_test_file`.
// Subsequently, it initializes a mocked evolution component
// `mock_klein_gordon_characteristic_evolution<Metavariables>` defined above
// that has the same `initialize_action_list` as
// `KleinGordonCharacteristicEvolution`. The function then goes through the
// action list and finally checks whether the tags are in the expected state.
template <typename Generator>
void test_klein_gordon_cce_initialization(const gsl::not_null<Generator*> gen) {
  using component = mock_klein_gordon_characteristic_evolution<metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;

  const std::string filename =
      "InitializeKleinGordonCharacteristicEvolutionTest_CceR0100.h5";

  // create the test file, because on initialization the manager will need to
  // get basic data out of the file
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
  TestHelpers::write_scalar_tensor_test_file(solution, filename, target_time,
                                             extraction_radius, frequency,
                                             amplitude, l_max);

  const double start_time = value_dist(*gen);
  const double target_step_size = 0.01 * value_dist(*gen);
  const size_t scri_plus_interpolation_order = 3;

  // tests start here
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {start_time, l_max, number_of_radial_points}};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0, target_step_size * 0.75,
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<::TimeSteppers::AdamsBashforth>(3)),
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(),
      target_step_size, scri_plus_interpolation_order);

  // go through the action list
  for (size_t i = 0; i < 6; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);

  // The tensor part:
  //
  // check that the tags are in the expected state (here we just make
  // sure it has the right size - it shouldn't have been written to yet).
  // This part is the same as `Test_InitializeCharacteristicEvolution.cpp`
  TestHelpers::check_characteristic_initialization<component>(
      runner, start_time, target_step_size, l_max, number_of_radial_points);

  // then we repeat the tests for the scalar (Klein-Gordon) part
  const auto& kg_boundary_communication_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename metavariables::klein_gordon_boundary_communication_tags>>(
      runner, 0);
  CHECK(kg_boundary_communication_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& kg_gauge_boundary_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename metavariables::klein_gordon_gauge_boundary_tags>>(
      runner, 0);
  CHECK(kg_gauge_boundary_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& kg_scri_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename metavariables::klein_gordon_scri_tags>>(runner,
                                                                         0);
  CHECK(kg_scri_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& kg_pre_swsh_derivative_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename metavariables::klein_gordon_pre_swsh_derivative_tags>>(
      runner, 0);
  CHECK(kg_pre_swsh_derivative_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& kg_swsh_derivative_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename metavariables::klein_gordon_swsh_derivative_tags>>(runner,
                                                                      0);
  CHECK(kg_swsh_derivative_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& kg_transform_buffer_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename metavariables::klein_gordon_transform_buffer_tags>>(runner,
                                                                       0);
  CHECK(kg_transform_buffer_tags.number_of_grid_points() ==
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max) *
            number_of_radial_points);

  const auto& kg_source_tags = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename metavariables::klein_gordon_source_tags>>(
      runner, 0);
  CHECK(kg_source_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& kg_integrand_tags = ActionTesting::get_databox_tag<
      component, ::Tags::Variables<
                     typename metavariables::klein_gordon_cce_integrand_tags>>(
      runner, 0);
  CHECK(kg_integrand_tags.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.InitializeKleinGordonCce",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_klein_gordon_cce_initialization(make_not_null(&gen));
}
}  // namespace Cce
