// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindCommonHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
// Mock actions only exist so we don't need all other action down the line. In
// this test we'll only check that actions are queued since each individual
// event is tested elsewhere, but we do check the dependencies
struct MockContributeVolumeData {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const observers::ObservationId& /*observation_id*/,
      const std::string& /*subfile_name*/,
      const Parallel::ArrayComponentId& /*array_component_id*/,
      ElementVolumeData&& /*received_volume_data*/,
      const std::optional<std::vector<char>>& /*serialized_functions_of_time*/,
      const std::optional<std::string>& dependency = std::nullopt) {
    CHECK(dependency == std::optional{"InterpolationTargetA"});
  }
};

template <typename Metavariables>
struct MockObserver {
  using component_being_mocked = observers::Observer<Metavariables>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeVolumeData>;
  using with_these_simple_actions = tmpl::list<MockContributeVolumeData>;
};

struct MockFindApparentHorizon {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<double>& /*incoming_time*/,
      const ElementId<3>& /*incoming_element_id*/,
      const ::Mesh<3>& /*incoming_mesh*/,
      Variables<ah::vars_to_interpolate_to_target<3, ::Frame::Grid>>&&
      /*incoming_vars_to_interpolate*/,
      const std::optional<std::string>& /*dependency*/,
      const bool /*source_vars_have_already_been_received*/ = false) {}
};

struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;

  using frame = ::Frame::Grid;

  // Don't need callbacks
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "MockHorizonMetavars"; }
};

template <typename Metavariables>
struct MockHorizonComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      ah::Component<Metavariables, MockHorizonMetavars>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>, ah::Tags::BlocksForHorizonFind>;
  using mutable_global_cache_tags =
      tmpl::list<ah::Tags::PreviousSurface<MockHorizonMetavars>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<ah::FindApparentHorizon<MockHorizonMetavars>>;
  using with_these_simple_actions = tmpl::list<MockFindApparentHorizon>;
};

template <typename Metavariables>
struct MockElement {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<
      db::AddSimpleTags<::ah::source_vars<Metavariables::volume_dim>>>;
};

struct MockMetavariables {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<MockObserver<MockMetavariables>,
                                    MockHorizonComponent<MockMetavariables>,
                                    MockElement<MockMetavariables>>;
};

void common_horizon_event() {
  (void)MockHorizonMetavars::destination;
  ::domain::creators::register_derived_with_charm();
  using metavars = MockMetavariables;
  constexpr size_t Dim = metavars::volume_dim;
  const ElementId<Dim> element_id(0);
  const Element<Dim> element{element_id, {}};

  using obs_component = MockObserver<metavars>;
  using horizon_component = MockHorizonComponent<metavars>;
  using elem_component = MockElement<metavars>;

  const ::domain::creators::Brick brick{
      {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {0, 0, 0}, {5, 5, 5}};
  const auto block_names = brick.block_names();
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {brick.create_domain(),
       std::unordered_map<std::string, std::unordered_set<std::string>>{
           {"MockHorizonMetavars", {block_names.begin(), block_names.end()}}}},
      {ah::Storage::LockedPreviousSurface<Frame::Grid>{}}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_group_component<obs_component>(make_not_null(&runner));
  ActionTesting::emplace_array_component<horizon_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::emplace_component<elem_component>(make_not_null(&runner),
                                                   element_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const auto check_results = [&runner,
                              &element_id](const size_t num_queued_actions) {
    CHECK(ActionTesting::is_simple_action_queue_empty<elem_component>(
        runner, element_id));
    CHECK(ActionTesting::number_of_queued_simple_actions<obs_component>(
              runner, 0) == num_queued_actions);
    CHECK(ActionTesting::number_of_queued_simple_actions<horizon_component>(
              runner, 0) == num_queued_actions);
  };

  // No events queued yet
  check_results(0);

  const Mesh<Dim> mesh(5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const double observation_time = 2.0;

  // Formerly, every point for every component of every tensor in vars were set
  // to a constant value of 1.0. But this leads to a metric that is not
  // invertible, causing a floating-point exception. The following
  // guarantees an invertible metric by supplying a flat spacetime instead.
  // Since pi and phi are zero for Minkowski, it's actually less code to just
  // manually set spacetime metric here than to get the analytic solution and
  // call all the functions to assemble spacetime_metric, pi, phi.
  Variables<ah::source_vars<Dim>> vars(mesh.number_of_grid_points(), 0.0);
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(vars);
  get<0, 0>(spacetime_metric) = DataVector(mesh.number_of_grid_points(), -1.0);
  for (size_t spatial_index = 0; spatial_index < Dim; ++spatial_index) {
    spacetime_metric.get(spatial_index + 1, spatial_index + 1) =
        DataVector(mesh.number_of_grid_points(), 1.0);
  }
  auto& cache = ActionTesting::cache<elem_component>(runner, element_id);

  const LinkedMessageId<double> temporal_id{observation_time, std::nullopt};
  const ::Event::ObservationValue observation_value{"FindCommonHorizon",
                                                    observation_time};

  // Actual coords don't matter
  const auto logical_coords = logical_coordinates(mesh);
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavars>,
      Parallel::Tags::GlobalCache<metavars>, MockHorizonMetavars::time_tag,
      Tags::Time, ::Events::Tags::ObserverMesh<Dim>,
      ::domain::Tags::Coordinates<3, ::Frame::Inertial>,
      ::Tags::Variables<typename decltype(vars)::tags_list>>>(
      metavars{}, &cache, temporal_id, observation_time, mesh,
      tnsr::I<DataVector, 3, ::Frame::Inertial>{
          std::array{logical_coords[0], logical_coords[1], logical_coords[2]}},
      vars);

  using FindCommonHorizon = ah::Events::FindCommonHorizon<
      MockHorizonMetavars,
      tmpl::push_back<ah::source_vars<Dim>,
                      ::domain::Tags::Coordinates<3, ::Frame::Inertial>>>;

  const FindCommonHorizon find_common_horizon{"SubfileName",
                                              FloatingPointType::Double,
                                              {FloatingPointType::Double},
                                              {"Pi"}};

  CHECK(find_common_horizon.needs_evolved_variables());
  CHECK(find_common_horizon.is_ready(cache, element_id,
                                     std::add_pointer_t<elem_component>{}));

  // Only compute tags for cache items necessary for observation box since this
  // test just puts the tags in the regular box
  auto obs_box = make_observation_box<tmpl::list<
      Parallel::Tags::FromGlobalCache<::domain::Tags::Domain<Dim>, metavars>>>(
      make_not_null(&box));
  find_common_horizon(obs_box, mesh, element, cache, element_id,
                      std::add_pointer_t<elem_component>{}, observation_value);

  // Since this event is a combination of two events, and those two events are
  // individually tested, here we only check we have the correct number of
  // queued actions on each component.
  check_results(1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FindCommonHorizon",
                  "[ApparentHorizonFinder][Unit]") {
  common_horizon_event();
}
