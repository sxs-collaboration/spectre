// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/Actions/WorldtubeBoundaryMocking.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

namespace {

template <typename Metavariables>
struct mock_analytic_worldtube_boundary {
  using initialize_action_list =
      tmpl::list<Actions::InitializeWorldtubeBoundary<
                     AnalyticWorldtubeBoundary<Metavariables>>,
                 Parallel::Actions::TerminatePhase>;
  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
    phase_dependent_action_list>;
};

struct H5Metavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_h5_worldtube_boundary<H5Metavariables>>;
  static constexpr bool evolve_ccm = false;
};

struct GhMetavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_gh_worldtube_boundary<GhMetavariables>>;
  static constexpr bool evolve_ccm = false;
};

struct AnalyticMetavariables {
  static constexpr bool local_time_stepping = false;
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_analytic_worldtube_boundary<AnalyticMetavariables>>;
};

template <typename Generator>
void test_h5_initialization(const gsl::not_null<Generator*> gen) {
  using component = mock_h5_worldtube_boundary<H5Metavariables>;
  const size_t l_max = 8;
  const size_t end_time = 100.0;
  const size_t start_time = 0.0;
  const double extraction_radius = 100.0;
  ActionTesting::MockRuntimeSystem<H5Metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<H5Metavariables>>{
          l_max, extraction_radius, end_time, start_time}};

  const size_t buffer_size = 8;
  const std::string filename = "InitializeWorldtubeBoundaryTest_CceR0100.h5";

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

  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      Tags::H5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          false, false, std::optional<double>{}));

  // this should run the initialization
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
  // check that the h5 data manager copied out of the databox has the correct
  // properties that we can examine without running the other actions
  const auto& data_manager =
      ActionTesting::get_databox_tag<component,
                                     Tags::H5WorldtubeBoundaryDataManager>(
          runner, 0);
  CHECK(data_manager.get_l_max() == l_max);
  const auto time_span = data_manager.get_time_span();
  CHECK(time_span.first == 0);
  CHECK(time_span.second == 0);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename H5Metavariables::cce_boundary_communication_tags>>(runner,
                                                                      0);

  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

void test_gh_initialization() {
  using component = mock_gh_worldtube_boundary<GhMetavariables>;
  const size_t l_max = 8;
  const double extraction_radius = 100.0;
  ActionTesting::MockRuntimeSystem<GhMetavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<GhMetavariables>>{
          l_max, extraction_radius, std::numeric_limits<double>::infinity(),
          0.0}};

  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      InterfaceManagers::GhLocalTimeStepping{
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u)},
      InterfaceManagers::GhLockstep{});

  // this should run the initialization
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  runner.set_phase(Parallel::Phase::Evolve);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename GhMetavariables::cce_boundary_communication_tags>>(runner,
                                                                      0);
  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
}

template <typename SolutionType>
void test_analytic_initialization() {
  using component = mock_analytic_worldtube_boundary<AnalyticMetavariables>;
  const size_t l_max = 8;
  const double extraction_radius = 20.0;
  register_classes_with_charm<TimeSteppers::Rk3HesthavenSsp>();
  ActionTesting::MockRuntimeSystem<AnalyticMetavariables> runner{
      {l_max, extraction_radius, 100.0, 0.0}};

  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      AnalyticBoundaryDataManager{12_st, extraction_radius,
                                  std::make_unique<SolutionType>()},
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<::TimeSteppers::Rk3HesthavenSsp>()));
  // this should run the initialization
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  runner.set_phase(Parallel::Phase::Evolve);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename AnalyticMetavariables::cce_boundary_communication_tags>>(
      runner, 0);
  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeWorldtubeBoundary",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_h5_initialization(make_not_null(&gen));
  test_gh_initialization();
  test_analytic_initialization<Solutions::RotatingSchwarzschild>();
  CHECK_THROWS_WITH(
      (test_analytic_initialization<Solutions::RobinsonTrautman>()),
      Catch::Matchers::ContainsSubstring(
          "Do not use RobinsonTrautman analytic solution with"));
}
}  // namespace Cce
