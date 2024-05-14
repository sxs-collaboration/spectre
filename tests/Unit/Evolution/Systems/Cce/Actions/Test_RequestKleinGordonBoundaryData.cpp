// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeKleinGordonVariables.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/KleinGordonCharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/Actions/WorldtubeBoundaryMocking.hpp"
#include "Helpers/Evolution/Systems/Cce/KleinGordonBoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"

namespace Cce {
namespace Actions {
namespace {
std::vector<double> times_requested;  // NOLINT
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
    times_requested.push_back(time.substep_time());
  }
};
}  // namespace
}  // namespace Actions

namespace {
template <typename Metavariables>
struct mock_kg_characteristic_evolution {
  using component_being_mocked =
      KleinGordonCharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

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

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<Actions::RequestBoundaryData<
                         KleinGordonH5WorldtubeBoundary<Metavariables>,
                         mock_kg_characteristic_evolution<Metavariables>>,
                     Actions::RequestNextBoundaryData<
                         KleinGordonH5WorldtubeBoundary<Metavariables>,
                         mock_kg_characteristic_evolution<Metavariables>>>>>;
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
                 mock_kg_characteristic_evolution<test_metavariables>>;
};

// This function tests that `Actions::RequestBoundaryData` and
// `Actions::RequestNextBoundaryData` can be successfully invoked by
// `KleinGordonCharacteristicEvolution`.
//
// The function begins by generating and storing some scalar and tensor data
// into an HDF5 file named `filename` using `write_scalar_tensor_test_file`.
// Subsequently, it initializes a mocked evolution component
// `mock_kg_characteristic_evolution<Metavariables>` and a mocked boundary
// component `mock_klein_gordon_h5_worldtube_boundary<Metavariables>`. Next, the
// function invokes `Actions::RequestBoundaryData` and
// `Actions::RequestNextBoundaryData`, with `BoundaryComputeAndSendToEvolution`
// replaced by `MockBoundaryComputeAndSendToEvolution`. Here
// `MockBoundaryComputeAndSendToEvolution` simply stores the time stamps
// (`times_requested`) that the evolution component needs. Finally, we check
// that `times_requested` agrees with what we expect.
template <typename Generator>
void test_klein_gordon_boundary_data(const gsl::not_null<Generator*> gen) {
  using evolution_component =
      mock_kg_characteristic_evolution<test_metavariables>;
  using worldtube_component =
      mock_klein_gordon_h5_worldtube_boundary<test_metavariables>;
  const size_t number_of_radial_points = 10;
  const size_t l_max = 8;

  const std::string filename = "KleinGordonBoundaryDataTest_CceR0100.h5";

  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const gr::Solutions::KerrSchild solution{mass, spin, center};

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
  const double end_time = start_time + 10 * target_step_size;
  const size_t buffer_size = 5;

  // tests start here
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      {l_max, extraction_radius,
       Tags::EndTimeFromFile::create_from_options(end_time, filename, false),
       start_time, number_of_radial_points}};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  // requested step size and slab size chosen to be sure that the
  // chosen step is a predictable value (not subject to roundoff
  // fluctuations in the generated value)
  ActionTesting::emplace_component<evolution_component>(
      &runner, 0, target_step_size * 0.75,
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
  CHECK(Actions::times_requested[1] == start_time + target_step_size * 0.75);
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.RequestKleinGoronBoundaryData",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_klein_gordon_boundary_data(make_not_null(&gen));
}
}  // namespace Cce
