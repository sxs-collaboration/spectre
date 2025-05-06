// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FindCommonHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
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
// event is tested elsewhere
struct MockContributeVolumeData {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& /*observation_id*/,
                    const std::string& /*subfile_name*/,
                    const Parallel::ArrayComponentId& /*array_component_id*/,
                    ElementVolumeData&& /*received_volume_data*/) {}
};

struct MockInterpolatorReceiveVolumeData {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, size_t VolumeDim>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const LinkedMessageId<double>& /*temporal_id*/,
      const ElementId<VolumeDim>& /*element_id*/,
      const ::Mesh<VolumeDim>& /*mesh*/,
      Variables<typename Metavariables::interpolator_source_vars>&& /*vars*/) {}
};

template <typename InterpolationTargetTag>
struct MockAddTemporalIdsToInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const LinkedMessageId<double>&
                    /*temporal_id*/,
                    std::optional<std::string> /*dependency*/) {}  // NOLINT
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

template <typename Metavariables>
struct MockInterpolator {
  using component_being_mocked = intrp::Interpolator<Metavariables>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<::intrp::Actions::InitializeInterpolator<
          metavariables::volume_dim,
          intrp::Tags::VolumeVarsInfo<Metavariables,
                                      ::Tags::TimeAndPrevious<0>>,
          intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>>;

  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolatorReceiveVolumeData<
          ::Tags::TimeAndPrevious<0>>>;
  using with_these_simple_actions =
      tmpl::list<MockInterpolatorReceiveVolumeData>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct MockInterpolationTarget {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::AddTemporalIdsToInterpolationTarget<
          typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions =
      tmpl::list<MockAddTemporalIdsToInterpolationTarget<
          typename Metavariables::InterpolationTargetA>>;
};

template <typename Metavariables>
struct MockElement {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<
      db::AddSimpleTags<::ah::source_vars<Metavariables::volume_dim>>>;
  using const_global_cache_tags =
      tmpl::list<::domain::Tags::Domain<metavariables::volume_dim>,
                 InterpTargetTestHelpers::Tags::BlocksForInterpolation>;
};

struct MockMetavariables {
  static constexpr size_t volume_dim = 3;
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeAndPrevious<0>;
    // Not the normal vars we'd interpolate but this is just a test
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                   gh::Tags::Pi<DataVector, volume_dim>,
                   gh::Tags::Phi<DataVector, volume_dim>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::ApparentHorizon<InterpolationTargetA,
                                               ::Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetA>>;
  };
  using interpolator_source_vars = ::ah::source_vars<volume_dim>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list = tmpl::list<
      MockObserver<MockMetavariables>, MockInterpolator<MockMetavariables>,
      MockInterpolationTarget<MockMetavariables, InterpolationTargetA>,
      MockElement<MockMetavariables>>;
};

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FindCommonHorizon",
                  "[ApparentHorizonFinder][Unit]") {
  ::domain::creators::register_derived_with_charm();
  using metavars = MockMetavariables;
  const ElementId<metavars::volume_dim> element_id(0);
  const ElementId<metavars::volume_dim> array_index(element_id);

  using obs_component = MockObserver<metavars>;
  using interp_component = MockInterpolator<metavars>;
  using interp_target_component =
      MockInterpolationTarget<metavars, metavars::InterpolationTargetA>;
  using elem_component = MockElement<metavars>;

  const ::domain::creators::Brick brick{
      {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {0, 0, 0}, {5, 5, 5}};
  const auto block_names = brick.block_names();
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {brick.create_domain(),
       std::unordered_map<std::string, std::unordered_set<std::string>>{
           {"InterpolationTargetA", {block_names.begin(), block_names.end()}}},
       ::Verbosity::Silent}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_group_component<obs_component>(make_not_null(&runner));
  ActionTesting::emplace_group_component<interp_component>(
      make_not_null(&runner));
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<interp_target_component>(
      make_not_null(&runner), 0);
  ActionTesting::emplace_component<elem_component>(make_not_null(&runner),
                                                   array_index);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const auto check_results = [&runner,
                              &element_id](const size_t num_queued_actions) {
    CHECK(ActionTesting::is_simple_action_queue_empty<elem_component>(
        runner, element_id));
    CHECK(ActionTesting::number_of_queued_simple_actions<obs_component>(
              runner, 0) == num_queued_actions);
    CHECK(ActionTesting::number_of_queued_simple_actions<interp_component>(
              runner, 0) == num_queued_actions);
    CHECK(
        ActionTesting::number_of_queued_simple_actions<interp_target_component>(
            runner, 0) == num_queued_actions);
  };

  // No events queued yet
  check_results(0);

  const Mesh<metavars::volume_dim> mesh(5, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto);
  const double observation_time = 2.0;
  const Variables<metavars::interpolator_source_vars> vars(
      mesh.number_of_grid_points(), 1.0);
  auto& cache = ActionTesting::cache<elem_component>(runner, array_index);

  const LinkedMessageId<double> temporal_id{observation_time, std::nullopt};
  const ::Event::ObservationValue observation_value{"FindCommonHorizon",
                                                    observation_time};

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavars>,
      Parallel::Tags::GlobalCacheImpl<metavars>,
      metavars::InterpolationTargetA::temporal_id, ::Tags::Time,
      ::Events::Tags::ObserverMesh<metavars::volume_dim>,
      ::Tags::Variables<typename decltype(vars)::tags_list>>>(
      metavars{}, &cache, temporal_id, observation_time, mesh, vars);

  using FindCommonHorizon = ah::Events::FindCommonHorizon<
      metavars::volume_dim, typename metavars::InterpolationTargetA,
      typename metavars::interpolator_source_vars,
      typename metavars::interpolator_source_vars>;

  const FindCommonHorizon find_common_horizon{};

  CHECK(find_common_horizon.needs_evolved_variables());
  CHECK(find_common_horizon.is_ready(cache, 0,
                                     std::add_pointer_t<elem_component>{}));

  // Only compute tags for cache items necessary for observation box since this
  // test just puts the tags in the regular box
  auto obs_box =
      make_observation_box<tmpl::list<Parallel::Tags::FromGlobalCache<
          ::domain::Tags::Domain<metavars::volume_dim>>>>(make_not_null(&box));
  find_common_horizon(obs_box, mesh, cache, array_index,
                      std::add_pointer_t<elem_component>{}, observation_value);

  // Since this event is a combination of two events, and those two events are
  // individually tested, here we only check we have the correct number of
  // queued actions on each component.
  check_results(1);
}
}  // namespace
