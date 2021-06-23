// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
template <typename Metavariables>
struct mock_analytic_worldtube_boundary {
  using component_being_mocked = AnalyticWorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list =
      tmpl::list<::Actions::SetupDataBox,
                 Actions::InitializeWorldtubeBoundary<
                     AnalyticWorldtubeBoundary<Metavariables>>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>>;
};

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
          typename Metavariables::scri_values_to_observe,
          typename Metavariables::cce_boundary_component>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<Actions::RequestBoundaryData<
                         AnalyticWorldtubeBoundary<Metavariables>,
                         mock_characteristic_evolution<Metavariables>>,
                     Actions::ReceiveWorldtubeData<Metavariables>,
                     Actions::RequestNextBoundaryData<
                         AnalyticWorldtubeBoundary<Metavariables>,
                         mock_characteristic_evolution<Metavariables>>>>>;
};

struct test_metavariables {
  using evolved_swsh_tag = Tags::BondiJ;
  static constexpr bool local_time_stepping = false;
  using evolved_swsh_dt_tag = Tags::BondiH;
  using evolved_coordinates_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::CauchyCartesianCoords, Tags::InertialRetardedTime>>;
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using cce_boundary_component = AnalyticWorldtubeBoundary<test_metavariables>;
  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Tags::BondiR, Tags::DuRDividedByR, Tags::BondiJ,
                     Tags::Dr<Tags::BondiJ>, Tags::BondiBeta, Tags::BondiQ,
                     Tags::BondiU, Tags::BondiW, Tags::BondiH>,
          tmpl::bind<Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Tags::BondiUAtScri, Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega,
      Tags::Du<Tags::GaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>>>;

  using scri_values_to_observe = tmpl::list<>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      bondi_hypersurface_step_tags,
      tmpl::bind<integrand_terms_to_compute_for_bondi_variable, tmpl::_1>>>;
  using cce_integration_independent_tags = pre_computation_tags;
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
      tmpl::list<mock_analytic_worldtube_boundary<test_metavariables>,
                 mock_characteristic_evolution<test_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.AnalyticBoundaryCommunication",
    "[Unit][Cce]") {
  Parallel::register_classes_with_charm<
      Cce::Solutions::RotatingSchwarzschild>();
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  using evolution_component = mock_characteristic_evolution<test_metavariables>;
  using worldtube_component =
      mock_analytic_worldtube_boundary<test_metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;
  const double extraction_radius = 100.0;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  const double frequency = 0.1 * value_dist(gen);
  const double target_time = 50.0 * value_dist(gen);
  const double start_time = target_time;
  const double target_step_size = 0.01 * value_dist(gen);
  const double end_time = target_time + 10.0 * target_step_size;
  const size_t scri_plus_interpolation_order = 2_st;

  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<test_metavariables>>{
          l_max, end_time, start_time, number_of_radial_points}};

  const AnalyticBoundaryDataManager analytic_manager{
      l_max, extraction_radius,
      std::make_unique<Cce::Solutions::RotatingSchwarzschild>(extraction_radius,
                                                              1.0, frequency)};
  runner.set_phase(test_metavariables::Phase::Initialization);
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size,
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<::TimeSteppers::DormandPrince5>()),
      scri_plus_interpolation_order,
      serialize_and_deserialize(analytic_manager));
  // Serialize and deserialize to get around the lack of implicit copy
  // constructor.
  ActionTesting::emplace_component<worldtube_component>(
      &runner, 0, serialize_and_deserialize(analytic_manager),
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<::TimeSteppers::DormandPrince5>()));

  // Run the initializations
  for (size_t i = 0; i < 6; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }
  for (size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<worldtube_component>(make_not_null(&runner), 0);
  }
  runner.set_phase(test_metavariables::Phase::Evolve);

  // Execute the first request for boundary data
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  // Check that the receive action is appropriately not ready
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<evolution_component>(
      make_not_null(&runner), 0));

  // the response (`BoundaryComputeAndSendToEvolution`)
  ActionTesting::invoke_queued_simple_action<worldtube_component>(
      make_not_null(&runner), 0);
  // then `ReceiveWorldtubeBoundaryData` in the evolution
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      expected_boundary_variables{number_of_angular_points};
  analytic_manager.populate_hypersurface_boundary_data(
      make_not_null(&expected_boundary_variables), target_time);

  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&expected_boundary_variables, &runner](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs =
            ActionTesting::get_databox_tag<evolution_component, tag>(runner, 0);
        const auto& test_rhs = get<tag>(expected_boundary_variables);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
      });
}
}  // namespace Cce
