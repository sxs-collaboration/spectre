// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InsertInterpolationScriData.hpp"
#include "Evolution/Systems/Cce/Actions/ScriObserveInterpolated.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace Cce {
namespace {

template <typename Tag>
struct SetBoundaryValues {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<DbTags...>, Tag>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    ComplexDataVector set_values) noexcept {
    db::mutate<Tag>(
        make_not_null(&box), [&set_values](
                                 const gsl::not_null<db::item_type<Tag>*>
                                     spin_weighted_scalar_quantity) noexcept {
          get(*spin_weighted_scalar_quantity).data() = std::move(set_values);
        });
  }

  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<DbTags...>, Tag>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    DataVector set_values) noexcept {
    db::mutate<Tag>(
        make_not_null(&box), [&set_values](
                                 const gsl::not_null<db::item_type<Tag>*>
                                     scalar_quantity) noexcept {
          get(*scalar_quantity) = std::move(set_values);
        });
  }
};

template <typename Metavariables>
struct mock_observer {
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using const_global_cache_tags = tmpl::list<observers::Tags::VolumeFileName>;
  using initialize_action_list =
      tmpl::list<observers::Actions::InitializeWriter<Metavariables>>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve, tmpl::list<>>>;
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

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              Actions::InsertInterpolationScriData<Tags::News>,
              Actions::ScriObserveInterpolated<mock_observer<Metavariables>>,
              ::Actions::AdvanceTime>>>;
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
  using cce_scri_tags = tmpl::list<Cce::Tags::News>;

  using scri_values_to_observe = tmpl::list<Cce::Tags::News>;

  using observed_reduction_data_tags = tmpl::list<>;

  using component_list =
      tmpl::list<mock_characteristic_evolution<test_metavariables>,
                 mock_observer<test_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.ScriObserveInterpolated",
                  "[Unit][Cce]") {
  using evolution_component = mock_characteristic_evolution<test_metavariables>;
  using observation_component = mock_observer<test_metavariables>;

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};

  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;
  const size_t observation_l_max = 4;
  const size_t scri_output_density = 1;
  const std::string filename = "ScriObserveInterpolatedTest_CceVolumeOutput";

  const double start_time = 0.0;
  const double target_step_size = 0.1;
  const size_t scri_interpolation_size = 3;

  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {filename, l_max, number_of_radial_points,
       std::make_unique<::TimeSteppers::RungeKutta3>(), start_time,
       scri_output_density, observation_l_max}};

  runner.set_phase(test_metavariables::Phase::Initialization);
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size, scri_interpolation_size);
  if (file_system::check_if_file_exists(filename + "0.h5")) {
    file_system::rm(filename + "0.h5", true);
  }

  ActionTesting::emplace_component<observation_component>(&runner, 0);

  // the initialization actions
  for (size_t i = 0; i < 4; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }
  ActionTesting::next_action<observation_component>(make_not_null(&runner), 0);
  runner.set_phase(test_metavariables::Phase::Evolve);

  // generate data that will be well behaved for the interpolation and the
  // decomposition to modes done in the observation routine
  UniformCustomDistribution<double> time_dist{-0.01, 0.01};

  const double linear_coefficient = value_dist(gen) * 0.1;
  const double quadratic_coefficient = value_dist(gen) * 0.1;
  const size_t vector_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t data_points = 30;

  // random vector based on modes
  // Generate data uniform in r with all angular modes
  SpinWeighted<ComplexModalVector, -2> generated_modes;
  generated_modes.data() = make_with_random_values<ComplexModalVector>(
      make_not_null(&gen), make_not_null(&coefficient_distribution),
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max));
  for (const auto& mode : Spectral::Swsh::cached_coefficients_metadata(l_max)) {
    if (mode.l < 2) {
      generated_modes.data()[mode.transform_of_real_part_offset] = 0.0;
      generated_modes.data()[mode.transform_of_imag_part_offset] = 0.0;
    }
    if (mode.m == 0) {
      generated_modes.data()[mode.transform_of_real_part_offset] =
          real(generated_modes.data()[mode.transform_of_real_part_offset]);
      generated_modes.data()[mode.transform_of_imag_part_offset] =
          real(generated_modes.data()[mode.transform_of_imag_part_offset]);
    }
  }
  const SpinWeighted<ComplexDataVector, -2> random_vector =
      Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);

  for (size_t i = 0; i < 3 * data_points; ++i) {
    // this will give random times that are nonetheless guaranteed to be
    // monotonically increasing
    const DataVector time_vector =
        i * 0.1 / 3.0 +
        make_with_random_values<DataVector>(
            make_not_null(&gen), make_not_null(&time_dist), vector_size);
    ActionTesting::simple_action<evolution_component,
                                 SetBoundaryValues<Tags::InertialRetardedTime>>(
        make_not_null(&runner), 0, time_vector);
    ActionTesting::simple_action<evolution_component,
                                 SetBoundaryValues<Tags::News>>(
        make_not_null(&runner), 0,
        random_vector.data() * (1.0 + linear_coefficient * (time_vector) +
                                quadratic_coefficient * square(time_vector)));

    // should put the data we just set into the interpolator
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
    // should process the interpolation and write to file
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
    while (not ActionTesting::is_threaded_action_queue_empty<
           observation_component>(runner, 0)) {
      ActionTesting::invoke_queued_threaded_action<observation_component>(
          make_not_null(&runner), 0);
    }
  }
  const auto& interpolation_manager = ActionTesting::get_databox_tag<
      evolution_component,
      Tags::InterpolationManager<ComplexDataVector, Tags::News>>(runner, 0);
  // we won't have dropped all of the volume data or gotten through all of the
  // data, because the last few points will not be sufficiently centered on the
  // interpolation stencil
  CHECK(interpolation_manager.number_of_data_points() < 10);
  CHECK(interpolation_manager.number_of_target_times() < 8);

  // most of the interpolations should be written now, so we read in the file
  // and check that they are as expected.

  // scoped to close the file
  {
    h5::H5File<h5::AccessType::ReadOnly> read_file{filename + "0.h5"};
    const auto& dataset = read_file.get<h5::Dat>("/News");
    const Matrix data_matrix = dataset.get_data();
    CHECK(data_matrix.rows() > 20);
    const auto expected_goldberg_modes =
        Spectral::Swsh::libsharp_to_goldberg_modes(generated_modes, l_max);

    Approx interpolation_approx =
        Approx::custom()
            .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
            .scale(1.0);
    // skip the first time because the extrapolation will make that value
    // unreliable
    for (size_t i = 1; i < data_matrix.rows(); ++i) {
      for (size_t j = 0; j < square(observation_l_max + 1); ++j) {
        CHECK(data_matrix(i, 2 * j + 1) ==
              interpolation_approx(
                  real(expected_goldberg_modes.data()[j] *
                       (1.0 + linear_coefficient * data_matrix(i, 0) +
                        quadratic_coefficient * square(data_matrix(i, 0))))));
      }
    }
  }
  if (file_system::check_if_file_exists(filename + "0.h5")) {
    file_system::rm(filename + "0.h5", true);
  }
}
}  // namespace Cce
