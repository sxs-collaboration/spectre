// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
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
      Variables<typename Metavariables::interpolator_source_vars>&&
          vars) noexcept {
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
      /*temporal_ids*/) noexcept {
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
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          Actions::SetupDataBox,
          ::intrp::Actions::InitializeInterpolator<
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
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;

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
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
};

struct MockMetavariables {
  struct InterpolatorTargetA {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target = tmpl::list<Tags::Lapse>;
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

  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.InterpolateEvent",
                  "[Unit]") {
  using metavars = MockMetavariables;
  const ElementId<metavars::volume_dim> element_id(2);
  const ElementId<metavars::volume_dim> array_index(element_id);

  using interp_component = mock_interpolator<metavars>;
  using interp_target_component =
      mock_interpolation_target<metavars, metavars::InterpolatorTargetA>;
  using elem_component = mock_element<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<interp_target_component>(&runner, 0);
  ActionTesting::next_action<interp_target_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_component<elem_component>(&runner, array_index);
  ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                             array_index);
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  const Mesh<metavars::volume_dim> mesh(5, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto);
  const double observation_time = 2.0;
  Variables<metavars::interpolator_source_vars> vars(
      mesh.number_of_grid_points());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  const auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavars>,
                        metavars::InterpolatorTargetA::temporal_id,
                        domain::Tags::Mesh<metavars::volume_dim>,
                        ::Tags::Variables<typename decltype(vars)::tags_list>>>(
      metavars{}, TimeStepId(true, 0, Slab(0., observation_time).end()), mesh,
      vars);

  metavars::event event{};

  CHECK(static_cast<const Event&>(event).is_ready(
      box, ActionTesting::cache<elem_component>(runner, array_index),
      array_index, std::add_pointer_t<elem_component>{}));
  CHECK(event.needs_evolved_variables());
  event.run(box, ActionTesting::cache<elem_component>(runner, array_index),
            array_index, std::add_pointer_t<elem_component>{});

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
  CHECK(results.temporal_id.substep_time().value() == observation_time);
  CHECK(results.element_id == element_id);
  CHECK(results.mesh == mesh);
  CHECK(results.vars == vars);
}

}  // namespace
