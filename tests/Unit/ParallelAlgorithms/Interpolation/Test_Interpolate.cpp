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

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"
// Reason for above pragma: If I include
// "DataStructures/DataBox/Prefixes.hpp", IWYU tells me to remove it.
// If I remove it, IWYU tells me to include it.

class DataVector;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp::Actions {
template <typename TemporalId>
struct InterpolatorReceiveVolumeData;
}  // namespace intrp::Actions

namespace {

namespace Tags {
struct Lapse : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

struct MockInterpolatorReceiveVolumeData {
  struct Results {
    // Hardcode expected types here.
    ::TimeStepId temporal_id{};
    ElementId<3> element_id{};
    Mesh<3> mesh{};
    Variables<tmpl::list<Tags::Lapse>> vars{};
  };
  static Results results;

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, size_t VolumeDim>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ::TimeStepId& temporal_id,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&& vars) {
    results.temporal_id = temporal_id;
    results.element_id = element_id;
    results.mesh = mesh;
    results.vars = vars;
  }
};

MockInterpolatorReceiveVolumeData::Results
    MockInterpolatorReceiveVolumeData::results{};

size_t called_mock_add_temporal_ids_to_interpolation_target = 0;
template <typename InterpolationTargetTag>
struct MockAddTemporalIdsToInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      std::vector<
          typename Metavariables::InterpolatorTargetA::temporal_id::type>&&
      /*temporal_ids*/) {
    // We are not testing this Action here.
    // Do nothing except make sure it is called once.
    ++called_mock_add_temporal_ids_to_interpolation_target;
  }
};

template <typename Metavariables>
struct mock_interpolator {
  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using replace_these_simple_actions = tmpl::list<
      intrp::Actions::InterpolatorReceiveVolumeData<::Tags::TimeStepId>>;
  using with_these_simple_actions =
      tmpl::list<MockInterpolatorReceiveVolumeData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<::intrp::Actions::InitializeInterpolator<
          intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
          intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator<
          intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
          intrp::Tags::InterpolatedVarsHolders<Metavariables>>::
          return_tag_list>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::AddTemporalIdsToInterpolationTarget<
          typename Metavariables::InterpolatorTargetA>>;
  using with_these_simple_actions =
      tmpl::list<MockAddTemporalIdsToInterpolationTarget<
          typename Metavariables::InterpolatorTargetA>>;
};

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
};

struct MockMetavariables {
  struct InterpolatorTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target = tmpl::list<Tags::Lapse>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::LineSegment<InterpolatorTargetA, 3,
                                           Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tmpl::list<>,
                                                     InterpolatorTargetA>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<Tags::Lapse>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<
      mock_interpolator<MockMetavariables>,
      mock_interpolation_target<MockMetavariables, InterpolatorTargetA>,
      mock_element<MockMetavariables>>;

  using event = intrp::Events::Interpolate<volume_dim, InterpolatorTargetA,
                                           interpolator_source_vars>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.InterpolateEvent",
                  "[Unit]") {
  ::domain::FunctionsOfTime::register_derived_with_charm();
  using metavars = MockMetavariables;
  const ElementId<metavars::volume_dim> element_id(2);
  const ElementId<metavars::volume_dim> array_index(element_id);

  using interp_component = mock_interpolator<metavars>;
  using interp_target_component =
      mock_interpolation_target<metavars, metavars::InterpolatorTargetA>;
  using elem_component = mock_element<metavars>;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  const double initial_expr_time = 0.1;
  const std::string name{"FunctionToCheck"};
  functions_of_time[name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array<DataVector, 1>{{DataVector{1, 0.0}}},
          initial_expr_time);
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {std::move(functions_of_time)}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<interp_target_component>(&runner, 0);
  ActionTesting::next_action<interp_target_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_component<elem_component>(&runner, array_index);
  ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                             array_index);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const Mesh<metavars::volume_dim> mesh(5, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto);
  const double observation_time = 2.0;
  Variables<metavars::interpolator_source_vars> vars(
      mesh.number_of_grid_points());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);
  auto& cache = ActionTesting::cache<elem_component>(runner, array_index);

  const auto check_results = [&array_index, &element_id, &mesh,
                              &observation_time, &runner, &vars]() {
    // Invoke all actions
    runner.invoke_queued_simple_action<interp_component>(0);
    runner.invoke_queued_simple_action<interp_target_component>(0);

    // No more queued simple actions.
    CHECK(runner.is_simple_action_queue_empty<interp_component>(0));
    CHECK(runner.is_simple_action_queue_empty<interp_target_component>(0));
    CHECK(runner.is_simple_action_queue_empty<elem_component>(array_index));

    // Make sure MockAddTemporalIdsToInterpolationTarget was called once.
    CHECK(called_mock_add_temporal_ids_to_interpolation_target == 1);

    const auto& results = MockInterpolatorReceiveVolumeData::results;
    CHECK(results.temporal_id.substep_time() == observation_time);
    CHECK(results.element_id == element_id);
    CHECK(results.mesh == mesh);
    CHECK(results.vars == vars);
  };

  const TimeStepId temporal_id(true, 0, Slab(0., observation_time).end());
  const double invalid_time = 0.2;

  intrp::interpolate<MockMetavariables::InterpolatorTargetA>(
      temporal_id, mesh, cache, array_index, get<Tags::Lapse>(vars));

  check_results();

  called_mock_add_temporal_ids_to_interpolation_target = 0;
  MockInterpolatorReceiveVolumeData::results = {};

  // Test the event version
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavars>,
                        metavars::InterpolatorTargetA::temporal_id,
                        ::Tags::Time, domain::Tags::Mesh<metavars::volume_dim>,
                        ::Tags::Variables<typename decltype(vars)::tags_list>>>(
      metavars{}, temporal_id, invalid_time, mesh, vars);

  metavars::event event{};

  // Functions of time aren't ready yet.
  CHECK_FALSE(static_cast<const Event&>(event).is_ready(
      box, cache, array_index, std::add_pointer_t<elem_component>{}));

  // Update time in box and functions of time
  db::mutate<::Tags::Time>([](gsl::not_null<double*> time) { *time = 1.0; },
                           make_not_null(&box));
  Parallel::mutate<domain::Tags::FunctionsOfTime,
                   control_system::UpdateFunctionOfTime>(
      cache, name, initial_expr_time, DataVector{1, 0.0}, observation_time);

  // Now everything should be ready
  CHECK(static_cast<const Event&>(event).is_ready(
      box, cache, array_index, std::add_pointer_t<elem_component>{}));
  CHECK(event.needs_evolved_variables());

  event.run(
      make_observation_box<
          typename metavars::event::compute_tags_for_observation_box>(box),
      cache, array_index, std::add_pointer_t<elem_component>{});

  check_results();
}

}  // namespace
