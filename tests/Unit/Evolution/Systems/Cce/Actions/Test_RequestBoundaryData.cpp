// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/Actions/WorldtubeBoundaryMocking.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {
namespace {
std::vector<double> times_requested;
template <typename BoundaryComponent, typename EvolutionComponent>
struct MockBoundaryComputeAndSendToEvolution {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<std::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const TimeStepId& time) {
    times_requested.push_back(time.substep_time().value());
  }
};
}  // namespace
}  // namespace Actions

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
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<Actions::RequestBoundaryData<
                         H5WorldtubeBoundary<Metavariables>,
                         mock_characteristic_evolution<Metavariables>>,
                     Actions::RequestNextBoundaryData<
                         H5WorldtubeBoundary<Metavariables>,
                         mock_characteristic_evolution<Metavariables>>>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct test_metavariables {
  using evolved_swsh_tag = Tags::BondiJ;
  using evolved_swsh_dt_tag = Tags::BondiH;
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
      Tags::PartiallyFlatGaugeOmega, Tags::Du<Tags::PartiallyFlatGaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>>>;

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
  using cce_angular_coordinate_tags = tmpl::list<Tags::CauchyAngularCoords>;
  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::ScriPlusFactor<Cce::Tags::Psi4>>;

  using component_list =
      tmpl::list<mock_h5_worldtube_boundary<test_metavariables>,
                 mock_characteristic_evolution<test_metavariables>>;

  static constexpr bool uses_partially_flat_cartesian_coordinates = false;

  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.RequestBoundaryData",
                  "[Unit][Cce]") {
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  using evolution_component = mock_characteristic_evolution<test_metavariables>;
  using worldtube_component = mock_h5_worldtube_boundary<test_metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;

  const std::string filename = "BoundaryDataTest_CceR0100.h5";
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
  const gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;
  const double frequency = 0.1 * value_dist(gen);
  const double amplitude = 0.1 * value_dist(gen);
  const double target_time = 50.0 * value_dist(gen);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  // create the test file, because on initialization the manager will need
  // to get basic data out of the file
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  const double start_time = target_time;
  const double target_step_size = 0.01 * value_dist(gen);
  const double end_time = start_time + 10 * target_step_size;
  const size_t buffer_size = 5;
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {l_max,
       Tags::EndTimeFromFile::create_from_options(end_time, filename, false),
       start_time, number_of_radial_points}};

  ActionTesting::set_phase(make_not_null(&runner),
                           test_metavariables::Phase::Initialization);
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size, false,
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<::TimeSteppers::RungeKutta3>()));
  ActionTesting::emplace_component<worldtube_component>(
      &runner, 0,
      Tags::H5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3_st,
                                                                       4_st),
          false, false, std::optional<double>{}));

  // this should run the initializations
  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }
  for(size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<worldtube_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           test_metavariables::Phase::Evolve);

  // the first request
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  // the first response
  ActionTesting::invoke_queued_simple_action<worldtube_component>(
      make_not_null(&runner), 0);
  CHECK(Actions::times_requested.size() == 1);
  CHECK(Actions::times_requested[0] == start_time);

  // the second request (the next substep)
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  // the second response
  ActionTesting::invoke_queued_simple_action<worldtube_component>(
      make_not_null(&runner), 0);
  CHECK(Actions::times_requested.size() == 2);
  CHECK(Actions::times_requested[1] == start_time + target_step_size);
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace Cce
