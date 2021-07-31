// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/algorithm/string/erase.hpp>
#include <boost/algorithm/string/replace.hpp>
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
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace Cce {
namespace {

template <typename Tag>
struct SetBoundaryValues {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<DbTags...>, Tag>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    ComplexDataVector set_values) noexcept {
    db::mutate<Tag>(
        make_not_null(&box), [&set_values](
                                 const gsl::not_null<typename Tag::type*>
                                     spin_weighted_scalar_quantity) noexcept {
          get(*spin_weighted_scalar_quantity).data() = std::move(set_values);
        });
  }

  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<DbTags...>, Tag>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    DataVector set_values) noexcept {
    db::mutate<Tag>(
        make_not_null(&box), [&set_values](
                                 const gsl::not_null<typename Tag::type*>
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

  using const_global_cache_tags =
      tmpl::list<observers::Tags::VolumeFileName, Tags::ObservationLMax>;
  using initialize_action_list =
      tmpl::list<::Actions::SetupDataBox,
                 observers::Actions::InitializeWriter<Metavariables>>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
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

  using initialize_action_list = tmpl::list<
      ::Actions::SetupDataBox,
      Actions::InitializeCharacteristicEvolutionVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionTime<
          typename Metavariables::evolved_coordinates_variables_tag,
          typename Metavariables::evolved_swsh_tag, false>,
      Actions::InitializeCharacteristicEvolutionScri<
          typename Metavariables::scri_values_to_observe,
          typename Metavariables::cce_boundary_component>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              tmpl::transform<
                  typename Metavariables::scri_values_to_observe,
                  tmpl::bind<Actions::InsertInterpolationScriData, tmpl::_1,
                             tmpl::pin<typename Metavariables::
                                           cce_boundary_component>>>,
              Actions::ScriObserveInterpolated<
                  mock_observer<Metavariables>,
                  typename Metavariables::cce_boundary_component>,
              ::Actions::AdvanceTime>>>;
};

struct test_metavariables {
  using evolved_swsh_tag = Tags::BondiJ;
  using evolved_swsh_dt_tag = Tags::BondiH;
  static constexpr bool local_time_stepping = false;
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

  using const_global_cache_tags = tmpl::list<Tags::SpecifiedStartTime>;
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
      tmpl::list<Tags::News, Tags::ScriPlus<Tags::Psi3>,
                 Tags::ScriPlus<Tags::Psi2>, Tags::ScriPlus<Tags::Psi1>,
                 Tags::ScriPlus<Tags::Psi0>,
                 Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>,
                 Tags::EthInertialRetardedTime>;

  using scri_values_to_observe =
      tmpl::list<Tags::News, Tags::ScriPlus<Tags::Psi3>,
                 Tags::ScriPlus<Tags::Psi2>, Tags::ScriPlus<Tags::Psi1>,
                 Tags::ScriPlus<Tags::Psi0>,
                 Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>,
                 Tags::EthInertialRetardedTime>;

  using observed_reduction_data_tags = tmpl::list<>;
  using cce_boundary_component =
      Cce::AnalyticWorldtubeBoundary<test_metavariables>;

  using component_list =
      tmpl::list<mock_characteristic_evolution<test_metavariables>,
                 mock_observer<test_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

template <typename TagToCalculate, typename... Tags>
ComplexDataVector compute_expected_field_from_pypp(
    const Variables<tmpl::list<Tags...>>& random_values,
    const double linear_coefficient, const double quadratic_coefficient,
    const double time, TagToCalculate /*meta*/) noexcept {
  const size_t size = random_values.number_of_grid_points();
  Scalar<DataVector> linear_coefficient_vector;
  get(linear_coefficient_vector) = DataVector{size, linear_coefficient};
  Scalar<DataVector> quadratic_coefficient_vector;
  get(quadratic_coefficient_vector) = DataVector{size, quadratic_coefficient};
  Scalar<DataVector> time_vector;
  get(time_vector) = DataVector{size, time};
  std::string tag_name = db::tag_name<TagToCalculate>();
  boost::algorithm::replace_all(tag_name, "(", "_");
  boost::algorithm::erase_all(tag_name, ")");
  return get(pypp::call<typename TagToCalculate::type>(
                 "ScriObserveInterpolated", "compute_" + tag_name,
                 linear_coefficient_vector, quadratic_coefficient_vector,
                 time_vector, get<Tags>(random_values)...))
      .data();
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.ScriObserveInterpolated",
                  "[Unit][Cce]") {
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  Parallel::register_classes_with_charm<
      Cce::Solutions::RotatingSchwarzschild>();
  Parallel::register_classes_with_charm<Cce::Solutions::TeukolskyWave>();
  using evolution_component = mock_characteristic_evolution<test_metavariables>;
  using observation_component = mock_observer<test_metavariables>;
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Cce/Actions/"};
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};

  const size_t number_of_radial_points = 6;
  const size_t l_max = 6;
  const size_t scri_output_density = 1;
  const std::string filename = "ScriObserveInterpolatedTest_CceVolumeOutput";

  const double start_time = 0.0;
  const double target_step_size = 0.1;
  const size_t scri_interpolation_size = 3;

  const double amplitude = 0.01 * value_dist(gen);
  const double duration = 50.0;
  const double extraction_radius = 100.0;
  Solutions::TeukolskyWave analytic_solution{extraction_radius, amplitude,
                                             duration};
  const AnalyticBoundaryDataManager analytic_manager{
    l_max, extraction_radius, analytic_solution.get_clone()};

  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {start_time, filename, l_max, l_max, number_of_radial_points,
       scri_output_density, false}};

  runner.set_phase(test_metavariables::Phase::Initialization);
  // Serialize and deserialize to get around the lack of implicit copy
  // constructor.
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size, false,
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<::TimeSteppers::RungeKutta3>()),
      scri_interpolation_size, serialize_and_deserialize(analytic_manager));
  if (file_system::check_if_file_exists(filename + "0.h5")) {
    file_system::rm(filename + "0.h5", true);
  }

  ActionTesting::emplace_component<observation_component>(&runner, 0);

  // the initialization actions
  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<observation_component>(make_not_null(&runner),
                                                      0);
  }
  runner.set_phase(test_metavariables::Phase::Evolve);

  // generate data that will be well behaved for the interpolation and the
  // decomposition to modes done in the observation routine
  UniformCustomDistribution<double> time_dist{-0.01, 0.01};

  const double linear_coefficient = value_dist(gen) * 0.1;
  const double quadratic_coefficient = value_dist(gen) * 0.1;
  const size_t vector_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t data_points = 30;

  Variables<typename test_metavariables::cce_scri_tags> random_scri_values{
      vector_size};
  tmpl::for_each<typename test_metavariables::cce_scri_tags>(
      [&l_max, &random_scri_values, &gen,
       &coefficient_distribution](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        SpinWeighted<ComplexModalVector, tag::type::type::spin> generated_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<tag::type::type::spin>(
            make_not_null(&generated_modes.data()), make_not_null(&gen),
            make_not_null(&coefficient_distribution), 1, l_max);
        Spectral::Swsh::inverse_swsh_transform(
            l_max, 1, make_not_null(&get(get<tag>(random_scri_values))),
            generated_modes);
      });

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
    tmpl::for_each<typename test_metavariables::cce_scri_tags>(
        [&runner, &random_scri_values, &linear_coefficient, &time_vector,
         &quadratic_coefficient](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          ActionTesting::simple_action<evolution_component,
                                       SetBoundaryValues<tag>>(
              make_not_null(&runner), 0,
              get(get<tag>(random_scri_values)).data() *
                  (1.0 + linear_coefficient * (time_vector) +
                   quadratic_coefficient * square(time_vector)));
        });

    // should put the data we just set into the interpolator
    for (size_t j = 0; j < tmpl::size<test_metavariables::cce_scri_tags>::value;
         ++j) {
      ActionTesting::next_action<evolution_component>(make_not_null(&runner),
                                                      0);
    }
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
    Approx interpolation_approx =
        Approx::custom()
            .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
            .scale(1.0);
    tmpl::for_each<typename test_metavariables::scri_values_to_observe>(
        [&random_scri_values, &linear_coefficient, &quadratic_coefficient,
         &l_max, &interpolation_approx, &read_file](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          const auto& dataset = read_file.get<h5::Dat>(
              "/" + Actions::detail::ScriOutput<tag>::name());
          const Matrix data_matrix = dataset.get_data();
          CHECK(data_matrix.rows() > 20);
          // skip the first time because the extrapolation will make that value
          // unreliable
          INFO(db::tag_name<tag>());
          for (size_t i = 1; i < data_matrix.rows(); ++i) {
            SpinWeighted<ComplexDataVector, tag::type::type::spin> expected;
            expected.data() = compute_expected_field_from_pypp(
                random_scri_values, linear_coefficient, quadratic_coefficient,
                data_matrix(i, 0), tag{});
            const auto expected_goldberg_modes =
                Spectral::Swsh::libsharp_to_goldberg_modes(
                    Spectral::Swsh::swsh_transform(l_max, 1, expected), l_max);
            for (size_t j = 0; j < square(l_max + 1); ++j) {
              CHECK(data_matrix(i, 2 * j + 1) ==
                    interpolation_approx(
                        real(expected_goldberg_modes.data()[j])));
              CHECK(data_matrix(i, 2 * j + 2) ==
                    interpolation_approx(
                        imag(expected_goldberg_modes.data()[j])));
            }
          }
        });
    const auto& dataset = read_file.get<h5::Dat>("/News_expected");
    const Matrix data_matrix = dataset.get_data();
    for (size_t i = 1; i < data_matrix.rows(); ++i) {
      const ComplexModalVector analytic_news_modes =
          Spectral::Swsh::libsharp_to_goldberg_modes(
              Spectral::Swsh::swsh_transform(
                  l_max, 1,
                  get(get<Tags::News>(analytic_solution.variables(
                      l_max, data_matrix(i, 0), tmpl::list<Tags::News>{})))),
              l_max)
              .data();
      CAPTURE(i);
      for (size_t j = 0; j < square(l_max + 1); ++j) {
        CHECK(data_matrix(i, 2 * j + 1) == real(analytic_news_modes.data()[j]));
        CHECK(data_matrix(i, 2 * j + 2) == imag(analytic_news_modes.data()[j]));
      }
    }
  }
  if (file_system::check_if_file_exists(filename + "0.h5")) {
    file_system::rm(filename + "0.h5", true);
  }
}
}  // namespace Cce
