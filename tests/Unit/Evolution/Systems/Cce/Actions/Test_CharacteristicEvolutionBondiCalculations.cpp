// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/CharacteristicEvolutionBondiCalculations.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

namespace {
template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list =
      tmpl::list<Actions::InitializeCharacteristicEvolution,
                 Actions::ReceiveWorldtubeData<Metavariables>,
                 ::Actions::MutateApply<InitializeJ<Tags::BoundaryValue>>,
                 ::Actions::MutateApply<InitializeGauge>,
                 ::Actions::MutateApply<GaugeUpdateAngularFromCartesian<
                     Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>,
                 ::Actions::MutateApply<GaugeUpdateJacobianFromCoordinates<
                     Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
                     Tags::CauchyCartesianCoords>>,
                 ::Actions::MutateApply<
                     GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>,
                 ::Actions::MutateApply<GaugeUpdateOmega>,
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
          tmpl::list<Actions::CalculateIntegrandInputsForTag<Tags::BondiBeta>,
                     Actions::PrecomputeGlobalCceDependencies>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct metavariables {
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

  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};

struct TestSendToEvolution {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const TimeStepId& time,
      const Variables<typename Metavariables::cce_boundary_communication_tags>&
          data_to_send) noexcept {
    Parallel::receive_data<Cce::ReceiveTags::BoundaryData<
        typename Metavariables::cce_boundary_communication_tags>>(
        Parallel::get_parallel_component<
            mock_characteristic_evolution<metavariables>>(cache),
        time, data_to_send, true);
  }
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.CharacteristicBondiCalculations",
    "[Unit][Cce]") {
  using component = mock_characteristic_evolution<metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<metavariables>>{
          std::make_unique<::TimeSteppers::RungeKutta3>(), l_max,
          number_of_radial_points}};

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

  // create the test file, because on initialization the manager will need
  // to get basic data out of the file
  const double start_time = target_time;
  const double end_time = std::numeric_limits<double>::quiet_NaN();
  const double target_step_size = 0.01 * value_dist(gen);

  runner.set_phase(metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0, start_time, start_time + target_step_size, target_step_size);

  // this should run the initialization
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  const size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  // manually create and place the boundary data in the box:
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

  using boundary_variables_tag = ::Tags::Variables<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>;
  using pre_swsh_derivative_tag_list =
      tmpl::append<pre_swsh_derivative_tags_to_compute_for_t<Tags::BondiBeta>,
                   tmpl::list<Tags::BondiJ, Tags::BondiBeta>>;

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  auto boundary_box = db::create<db::AddSimpleTags<
      boundary_variables_tag, ::Tags::Variables<pre_swsh_derivative_tag_list>,
      ::Tags::Variables<typename metavariables::cce_gauge_boundary_tags>,
      typename metavariables::evolved_coordinates_variables_tag,
      ::Tags::Variables<typename metavariables::cce_angular_coordinate_tags>,
      ::Tags::Variables<
          typename metavariables::cce_integration_independent_tags>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::LMax, Spectral::Swsh::Tags::NumberOfRadialPoints>>(
      db::item_type<boundary_variables_tag>{number_of_angular_points},
      Variables<pre_swsh_derivative_tag_list>{number_of_radial_points *
                                              number_of_angular_points},
      Variables<typename metavariables::cce_gauge_boundary_tags>{
          number_of_angular_points},
      db::item_type<typename metavariables::evolved_coordinates_variables_tag>{
          number_of_angular_points},
      Variables<typename metavariables::cce_angular_coordinate_tags>{
          number_of_angular_points},
      Variables<typename metavariables::cce_integration_independent_tags>{
          number_of_angular_points * number_of_radial_points},
      Spectral::Swsh::SwshInterpolator{}, l_max, number_of_radial_points);
  create_bondi_boundary_data(
      make_not_null(&boundary_box), spatial_metric_coefficients,
      dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
      shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
      lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
      extraction_radius, l_max);

  ActionTesting::simple_action<component, TestSendToEvolution>(
      make_not_null(&runner), 0,
      ActionTesting::get_databox_tag<component, ::Tags::TimeStepId>(runner, 0),
      db::get<::Tags::Variables<
          Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>>(
          boundary_box));

  // the rest of the initialization routine
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(metavariables::Phase::Evolve);

  // this should run the computation actions
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  // perform the expected transformations on the `boundary_box`
  db::mutate_apply<InitializeJ<Tags::BoundaryValue>>(
      make_not_null(&boundary_box));
  db::mutate_apply<InitializeGauge>(make_not_null(&boundary_box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&boundary_box));
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>>(make_not_null(&boundary_box));
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      make_not_null(&boundary_box));
  db::mutate_apply<GaugeUpdateOmega>(make_not_null(&boundary_box));

  tmpl::for_each<gauge_adjustments_setup_tags>([&boundary_box](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(
        make_not_null(&boundary_box));
  });
  mutate_all_precompute_cce_dependencies<Tags::EvolutionGaugeBoundaryValue>(
      make_not_null(&boundary_box));
  mutate_all_pre_swsh_derivatives_for_tag<Tags::BondiBeta>(
      make_not_null(&boundary_box));
  mutate_all_swsh_derivatives_for_tag<Tags::BondiBeta>(
      make_not_null(&boundary_box));

  using bondi_calculation_result_tags =
      tmpl::append<pre_computation_tags,
                   pre_swsh_derivative_tags_to_compute_for_t<Tags::BondiBeta>>;

  // check the main result tags for correctness
  tmpl::for_each<bondi_calculation_result_tags>(
      [&runner, &boundary_box](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(tag::name());
        const auto& test_lhs =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& test_rhs = db::get<tag>(boundary_box);
        CAPTURE(test_rhs);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
      });
}
}  // namespace Cce
