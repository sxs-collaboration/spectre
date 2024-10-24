// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>

#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Actions/InsertInterpolationScriData.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace Cce {
namespace {

std::unordered_map<std::string, std::vector<double>> written_modes;
struct MockWriteReductionDataRow {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& /*file_legend*/,
                    const std::tuple<std::vector<double>>& data_row) {
    written_modes[subfile_name] = std::get<0>(data_row);
  }
};

struct RotatingSchwarzschildWithNoninertialNews
    : public Cce::Solutions::RotatingSchwarzschild {
  using Cce::Solutions::RotatingSchwarzschild::RotatingSchwarzschild;
  // ignore warning for unused | operator
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(RotatingSchwarzschildWithNoninertialNews);
#pragma GCC diagnostic pop
  bool use_noninertial_news() const override { return true; }

  std::unique_ptr<Cce::Solutions::WorldtubeData> get_clone() const override {
    return std::make_unique<RotatingSchwarzschildWithNoninertialNews>(*this);
  }
  void pup(PUP::er& p) override {
    Cce::Solutions::RotatingSchwarzschild::pup(p);
  }
};

PUP::able::PUP_ID RotatingSchwarzschildWithNoninertialNews::my_PUP_ID = 0;

struct SetRandomBoundaryValues {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    MAKE_GENERATOR(gen);
    UniformCustomDistribution<double> value_dist{0.1, 0.5};
    tmpl::for_each<typename Metavariables::cce_scri_tags>(
        [&gen, &value_dist, &box](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              [&gen, &value_dist](
                  const gsl::not_null<typename tag::type*> scri_value) {
                fill_with_random_values(make_not_null(&get(*scri_value).data()),
                                        make_not_null(&gen),
                                        make_not_null(&value_dist));
              },
              make_not_null(&box));
        });

    db::mutate<Tags::InertialRetardedTime>(
        [&gen,
         &value_dist](const gsl::not_null<Scalar<DataVector>*> scri_value) {
          fill_with_random_values(make_not_null(&get(*scri_value)),
                                  make_not_null(&gen),
                                  make_not_null(&value_dist));
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables>
struct MockObserver {
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionDataRow>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionDataRow>;

  using const_global_cache_tags = tmpl::list<Tags::ObservationLMax>;
  using initialize_action_list =
      tmpl::list<observers::Actions::InitializeWriter<Metavariables>>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockCharacteristicEvolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list = tmpl::list<
      Actions::InitializeCharacteristicEvolutionVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionTime<
          typename Metavariables::evolved_coordinates_variables_tag,
          typename Metavariables::evolved_swsh_tags, false>,
      // advance the time so that the current `TimeStepId` is valid without
      // having to perform self-start.
      ::Actions::AdvanceTime,
      Actions::InitializeCharacteristicEvolutionScri<
          typename Metavariables::scri_values_to_observe,
          typename Metavariables::cce_boundary_component>,
      Parallel::Actions::TerminatePhase>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<
              SetRandomBoundaryValues,
              tmpl::transform<
                  typename Metavariables::scri_values_to_observe,
                  tmpl::bind<Actions::InsertInterpolationScriData, tmpl::_1,
                             tmpl::pin<typename Metavariables::
                                           cce_boundary_component>>>>>>;
};

struct test_metavariables {
  using evolved_swsh_tags = tmpl::list<Tags::BondiJ>;
  using evolved_swsh_dt_tags = tmpl::list<Tags::BondiH>;
  using cce_step_choosers = tmpl::list<>;
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

  using const_global_cache_tags = tmpl::list<Tags::SpecifiedStartTime>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, tmpl::list<>>>;
  };

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

  using observed_reduction_data_tags = tmpl::list<>;
  using cce_boundary_component =
      Cce::AnalyticWorldtubeBoundary<test_metavariables>;
  using ccm_psi0 = tmpl::list<
        Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>,
        Cce::Tags::BoundaryValue<Cce::Tags::Dlambda<Cce::Tags::Psi0Match>>>;

  using component_list =
      tmpl::list<MockCharacteristicEvolution<test_metavariables>,
                 MockObserver<test_metavariables>>;
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InsertInterpolationScriData",
    "[Unit][Cce]") {
  register_classes_with_charm<Cce::Solutions::RotatingSchwarzschild>();
  register_classes_with_charm<RotatingSchwarzschildWithNoninertialNews>();
  using evolution_component = MockCharacteristicEvolution<test_metavariables>;
  using observation_component = MockObserver<test_metavariables>;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;
  const size_t scri_output_density = 5;

  const double start_time = value_dist(gen);
  const double target_step_size = 0.01 * value_dist(gen);
  const size_t buffer_size = 5;
  const double extraction_radius = 100.0;

  const RotatingSchwarzschildWithNoninertialNews analytic_solution{
      extraction_radius, 1.0, 0.05};
  const AnalyticBoundaryDataManager analytic_manager{
      l_max, extraction_radius, analytic_solution.get_clone()};
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {start_time, l_max, l_max, number_of_radial_points, scri_output_density,
       true}};
  runner.set_phase(Parallel::Phase::Initialization);
  // Serialize and deserialize to get around the lack of implicit copy
  // constructor.
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size,
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<::TimeSteppers::AdamsBashforth>(3)),
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(),
      target_step_size, buffer_size,
      serialize_and_deserialize(analytic_manager));
  ActionTesting::emplace_component<MockObserver<test_metavariables>>(&runner,
                                                                     0);
  // the initialization actions
  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::next_action<evolution_component>(make_not_null(&runner), 0);
  }
  runner.set_phase(Parallel::Phase::Evolve);
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

  // check the output news against the news provided by the analytic solution
  ActionTesting::invoke_queued_threaded_action<observation_component>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<observation_component>(
      make_not_null(&runner), 0);
  const ComplexModalVector analytic_news_modes =
      Spectral::Swsh::libsharp_to_goldberg_modes(
          Spectral::Swsh::swsh_transform(
              l_max, 1,
              get(get<Tags::News>(analytic_solution.variables(
                  l_max, start_time, tmpl::list<Tags::News>{})))),
          l_max)
          .data();
  CHECK(written_modes["/Cce/News_Noninertial"].size() ==
        2 * square(l_max + 1) + 1);
  CHECK(written_modes["/Cce/News_Noninertial"][0] == start_time);
  CHECK(written_modes["/Cce/News_Noninertial_expected"][0] == start_time);
  for (size_t i = 0; i < square(l_max + 1); ++i) {
    CHECK(written_modes["/Cce/News_Noninertial_expected"][2 * i + 1] ==
          real(analytic_news_modes[i]));
    CHECK(written_modes["/Cce/News_Noninertial_expected"][2 * i + 2] ==
          imag(analytic_news_modes[i]));
  }
}
}  // namespace Cce
