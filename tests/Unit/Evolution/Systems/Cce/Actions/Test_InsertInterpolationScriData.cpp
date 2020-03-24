// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Actions/InsertInterpolationScriData.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

namespace Cce {
namespace {

struct SetRandomBoundaryValues {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    MAKE_GENERATOR(gen);
    UniformCustomDistribution<double> value_dist{0.1, 0.5};
    tmpl::for_each<typename Metavariables::cce_scri_tags>(
        [&gen, &value_dist, &box ](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              make_not_null(&box), [
                &gen, &value_dist
              ](const gsl::not_null<db::item_type<tag>*> scri_value) noexcept {
                fill_with_random_values(make_not_null(&get(*scri_value).data()),
                                        make_not_null(&gen),
                                        make_not_null(&value_dist));
              });
        });

    db::mutate<Tags::InertialRetardedTime>(
        make_not_null(&box),
        [&gen, &value_dist ](
            const gsl::not_null<db::item_type<Tags::InertialRetardedTime>*>
                scri_value) noexcept {
          fill_with_random_values(make_not_null(&get(*scri_value)),
                                  make_not_null(&gen),
                                  make_not_null(&value_dist));
        });
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list =
      tmpl::list<Actions::InitializeCharacteristicEvolutionVariables,
                 Actions::InitializeCharacteristicEvolutionTime,
                 Actions::InitializeCharacteristicEvolutionScri,
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
          tmpl::list<
              SetRandomBoundaryValues,
              tmpl::transform<typename Metavariables::scri_values_to_observe,
                              tmpl::bind<Actions::InsertInterpolationScriData,
                                         tmpl::_1>>>>>;
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
      Tags::BondiUAtScri, Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega,
      Tags::Du<Tags::GaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>>>;

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

  using scri_values_to_observe = tmpl::list<
      Cce::Tags::News,
      ::Tags::Multiplies<Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                         Cce::Tags::ScriPlus<Cce::Tags::Psi2>>,
      Cce::Tags::Du<
          Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
      ::Tags::Multiplies<Cce::Tags::Du<Cce::Tags::TimeIntegral<
                             Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
                         Cce::Tags::ScriPlusFactor<Cce::Tags::Psi4>>>;

  using component_list =
      tmpl::list<mock_characteristic_evolution<test_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InsertInterpolationScriData",
    "[Unit][Cce]") {
  using evolution_component = mock_characteristic_evolution<test_metavariables>;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;
  const size_t scri_output_density = 5;


  const double start_time = value_dist(gen);
  const double end_time = start_time + 1.0;
  const double target_step_size = 0.01 * value_dist(gen);
  const size_t buffer_size = 5;

  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {l_max, number_of_radial_points,
       std::make_unique<::TimeSteppers::RungeKutta3>(), start_time,
       Tags::EndTime::create_from_options(end_time, "unused"),
       scri_output_density}};

  runner.set_phase(test_metavariables::Phase::Initialization);
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size, buffer_size);

  // the initialization actions
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  runner.set_phase(test_metavariables::Phase::Evolve);
  // five steps to create then put the data in each of the interpolation
  // queues
  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }

  // check each of the interpolation queues to make sure that they've received
  // the data
  const auto& standard_interpolator = ActionTesting::get_databox_tag<
      evolution_component,
      Tags::InterpolationManager<ComplexDataVector, Tags::News>>(runner, 0);
  CHECK(standard_interpolator.number_of_data_points() == 1);
  CHECK(standard_interpolator.number_of_target_times() == 5);

  const auto& du_interpolator = ActionTesting::get_databox_tag<
      evolution_component,
      Tags::InterpolationManager<ComplexDataVector,
                                 Cce::Tags::Du<Cce::Tags::TimeIntegral<
                                     Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>>>(
      runner, 0);
  CHECK(du_interpolator.number_of_data_points() == 1);
  CHECK(du_interpolator.number_of_target_times() == 5);

  const auto& multiplies_interpolator = ActionTesting::get_databox_tag<
      evolution_component,
      Tags::InterpolationManager<
          ComplexDataVector,
          ::Tags::Multiplies<Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                             Cce::Tags::ScriPlus<Cce::Tags::Psi2>>>>(runner, 0);
  CHECK(multiplies_interpolator.number_of_data_points() == 1);
  CHECK(multiplies_interpolator.number_of_target_times() == 5);

  const auto& multiplies_du_interpolator = ActionTesting::get_databox_tag<
      evolution_component,
      Tags::InterpolationManager<
          ComplexDataVector,
          ::Tags::Multiplies<Cce::Tags::Du<Cce::Tags::TimeIntegral<
                                 Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
                             Cce::Tags::ScriPlusFactor<Cce::Tags::Psi4>>>>(
      runner, 0);
  CHECK(multiplies_du_interpolator.number_of_data_points() == 1);
  CHECK(multiplies_du_interpolator.number_of_target_times() == 5);
}
}  // namespace Cce
