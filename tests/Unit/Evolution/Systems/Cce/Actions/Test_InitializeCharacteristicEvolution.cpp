// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace {
template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list = tmpl::list<
      ::Actions::SetupDataBox,
      Actions::InitializeCharacteristicEvolutionVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionTime<
          typename Metavariables::evolved_coordinates_variables_tag,
          typename Metavariables::evolved_swsh_tag, false>,
      // advance the time so that the current `TimeStepId` is valid without
      // having to perform self-start.
      ::Actions::AdvanceTime,
      Actions::InitializeCharacteristicEvolutionScri<
          typename Metavariables::scri_values_to_observe, NoSuchType>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve, tmpl::list<>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct metavariables {
  using evolved_swsh_tag = Tags::BondiJ;
  using evolved_swsh_dt_tag = Tags::BondiH;
  using cce_step_choosers = tmpl::list<>;
  using evolved_coordinates_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::CauchyCartesianCoords, Tags::InertialRetardedTime>>;
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Tags::BondiR, Tags::DuRDividedByR, Tags::BondiJ,
                     Tags::Dr<Tags::BondiJ>, Tags::BondiBeta, Tags::BondiQ,
                     Tags::BondiU, Tags::BondiW, Tags::BondiH>,
          tmpl::bind<Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Tags::BondiUAtScri, Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Cce::Tags::CauchyGaugeC, Cce::Tags::CauchyGaugeD,
      Tags::PartiallyFlatGaugeOmega, Cce::Tags::CauchyGaugeOmega,
      Tags::Du<Tags::PartiallyFlatGaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::CauchyGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>>>;

  using const_global_cache_tags = tmpl::list<Tags::SpecifiedStartTime>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        StepChooser<StepChooserUse::LtsStep>,
        tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>,
                   StepChoosers::Increase<StepChooserUse::LtsStep>>>>;
  };

  using scri_values_to_observe = tmpl::list<>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      bondi_hypersurface_step_tags,
      tmpl::bind<integrand_terms_to_compute_for_bondi_variable, tmpl::_1>>>;
  using cce_integration_independent_tags =
      tmpl::append<pre_computation_tags, tmpl::list<Tags::DuRDividedByR>>;
  using cce_temporary_equations_tags =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          cce_integrand_tags, tmpl::bind<integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = all_transform_buffer_tags;
  using cce_swsh_derivative_tags = all_swsh_derivative_tags;
  using cce_angular_coordinate_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::PartiallyFlatAngularCoords>;
  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::ScriPlusFactor<Cce::Tags::Psi4>>;

  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeCharacteristicEvolution",
    "[Unit][Cce]") {
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  using component = mock_characteristic_evolution<metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;

  const std::string filename =
      "InitializeCharacteristicEvolutionTest_CceR0100.h5";
  // create the test file, because on initialization the manager will need to
  // get basic data out of the file

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(gen);
  const std::array<double, 3> spin{
      {value_dist(gen), value_dist(gen), value_dist(gen)}};
  const std::array<double, 3> center{
      {value_dist(gen), value_dist(gen), value_dist(gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;
  const double frequency = 0.1 * value_dist(gen);
  const double amplitude = 0.1 * value_dist(gen);
  const double target_time = 50.0 * value_dist(gen);
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  // create the test file, because on initialization the manager will need
  // to get basic data out of the file
  const double start_time = value_dist(gen);
  const double target_step_size = 0.01 * value_dist(gen);
  const size_t scri_plus_interpolation_order = 3;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {start_time, l_max, number_of_radial_points}};

  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0, target_step_size * 0.75, false,
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<::TimeSteppers::AdamsBashforthN>(3)),
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(),
      static_cast<std::unique_ptr<StepController>>(
          std::make_unique<StepControllers::BinaryFraction>()),
      target_step_size, scri_plus_interpolation_order);

  // this should run the initialization
  for (size_t i = 0; i < 6; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Evolve);

  // the tags inserted in the `EvolutionTags` step
  const auto& time_step_id =
      ActionTesting::get_databox_tag<component, ::Tags::TimeStepId>(runner, 0);
  CHECK(time_step_id.substep_time().value() == start_time);
  const auto& next_time_step_id =
      ActionTesting::get_databox_tag<component,
                                     ::Tags::Next<::Tags::TimeStepId>>(runner,
                                                                       0);
  CHECK(next_time_step_id.substep_time().value() ==
        approx(start_time + target_step_size * 0.75));
  const auto& time_step =
      ActionTesting::get_databox_tag<component, ::Tags::TimeStep>(runner, 0);
  CHECK(time_step.value() == approx(target_step_size * 0.75));
  const auto& coordinates_history = ActionTesting::get_databox_tag<
      component,
      ::Tags::HistoryEvolvedVariables<
          typename metavariables::evolved_coordinates_variables_tag>>(runner,
                                                                      0);
  CHECK(coordinates_history.size() == 0);
  const auto& evolved_swsh_history = ActionTesting::get_databox_tag<
      component, ::Tags::HistoryEvolvedVariables<::Tags::Variables<
                     tmpl::list<typename metavariables::evolved_swsh_tag>>>>(
      runner, 0);
  CHECK(evolved_swsh_history.size() == 0);

  // the tensor storage variables inserted during the `CharacteristicTags` step
  const auto& boundary_variables = ActionTesting::get_databox_tag<
      component, ::Tags::Variables<tmpl::append<
                     typename metavariables::cce_boundary_communication_tags,
                     typename metavariables::cce_gauge_boundary_tags>>>(runner,
                                                                        0);
  CHECK(boundary_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& coordinate_variables = ActionTesting::get_databox_tag<
      component, typename metavariables::evolved_coordinates_variables_tag>(
      runner, 0);
  CHECK(coordinate_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& dt_coordinate_variables = ActionTesting::get_databox_tag<
      component,
      db::add_tag_prefix<::Tags::dt, typename metavariables::
                                         evolved_coordinates_variables_tag>>(
      runner, 0);
  CHECK(dt_coordinate_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& angular_coordinates_variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename metavariables::cce_angular_coordinate_tags>>(
      runner, 0);
  CHECK(angular_coordinates_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  const auto& scri_variables = ActionTesting::get_databox_tag<
      component, ::Tags::Variables<typename metavariables::cce_scri_tags>>(
      runner, 0);
  CHECK(scri_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  const auto& volume_variables = ActionTesting::get_databox_tag<
      component, ::Tags::Variables<tmpl::append<
                     typename metavariables::cce_integrand_tags,
                     typename metavariables::cce_integration_independent_tags,
                     typename metavariables::cce_temporary_equations_tags>>>(
      runner, 0);
  CHECK(volume_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& evolved_swsh_variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<tmpl::list<typename metavariables::evolved_swsh_tag>>>(
      runner, 0);
  CHECK(evolved_swsh_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& evolved_swsh_dt_variables = ActionTesting::get_databox_tag<
      component, ::Tags::Variables<
                     tmpl::list<typename metavariables::evolved_swsh_dt_tag>>>(
      runner, 0);
  CHECK(evolved_swsh_dt_variables.number_of_grid_points() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
            number_of_radial_points);

  const auto& pre_swsh_derivatives_variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename metavariables::cce_pre_swsh_derivatives_tags>>(
      runner, 0);
  const Variables<typename metavariables::cce_pre_swsh_derivatives_tags>
      expected_zeroed_pre_swsh_derivatives{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
              number_of_radial_points,
          0.0};
  CHECK(pre_swsh_derivatives_variables == expected_zeroed_pre_swsh_derivatives);

  const auto& transform_buffer_variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename metavariables::cce_transform_buffer_tags>>(
      runner, 0);
  const Variables<typename metavariables::cce_transform_buffer_tags>
      expected_zeroed_transform_buffer{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max) *
              number_of_radial_points,
          0.0};
  CHECK(transform_buffer_variables == expected_zeroed_transform_buffer);

  const auto& swsh_derivatives_variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename metavariables::cce_swsh_derivative_tags>>(
      runner, 0);
  const Variables<typename metavariables::cce_swsh_derivative_tags>
      expected_zeroed_swsh_derivatives{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
              number_of_radial_points,
          0.0};
  CHECK(swsh_derivatives_variables == expected_zeroed_swsh_derivatives);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace Cce
