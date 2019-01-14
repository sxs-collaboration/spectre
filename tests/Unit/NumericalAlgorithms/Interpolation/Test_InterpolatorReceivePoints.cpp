// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars;
}  // namespace Actions
}  // namespace intrp
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
template <typename Metavariables>
struct InterpolatedVarsHolders;
template <typename Metavariables>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace {

size_t num_calls_of_target_receive_vars = 0;
template <typename InterpolationTargetTag>
struct MockInterpolationTargetReceiveVars {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const std::vector<db::item_type<::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>>&
      /*vars_src*/,
      const std::vector<std::vector<size_t>>& /*global_offsets*/) noexcept {
    // InterpolationTargetReceiveVars will not be called in this test,
    // because we are not supplying volume data (so try_to_interpolate
    // inside TryToInterpolate.hpp will not actually interpolate). However, the
    // compiler thinks that InterpolationTargetReceiveVars might be called, so
    // we mock it so that everything compiles.
    //
    // Note that try_to_interpolate and InterpolationTargetReceiveVars have
    // already been tested by other tests.

    // Here we increment a variable so that later we can
    // verify that this wasn't called.
    ++num_calls_of_target_receive_vars;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolationTargetReceiveVars<
          typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions =
      tmpl::list<MockInterpolationTargetReceiveVars<
          typename Metavariables::InterpolationTargetA>>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename intrp::Actions::InitializeInterpolator::
                                   template return_tag_list<Metavariables>>;
  using component_being_mocked = void;  // not needed.
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_source = tmpl::list<>;
  };
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ReceivePoints",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars>>;
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      mock_interpolator<metavars>>{});
  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  runner.simple_action<mock_interpolator<metavars>,
                       ::intrp::Actions::InitializeInterpolator>(0);

  // Make sure that we have one Element registered,
  // or else ReceivePoints will (correctly) do nothing because it
  // thinks it will never have any Elements to interpolate onto.
  runner.simple_action<mock_interpolator<metavars>,
                       ::intrp::Actions::RegisterElement>(0);

  const auto& box =
      runner.template algorithms<mock_interpolator<metavars>>()
          .at(0)
          .template get_databox<
              typename mock_interpolator<metavars>::initial_databox>();

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{7, 7}}, false);
  const auto domain = domain_creator.create_domain();
  const auto block_logical_coords = [&domain]() noexcept {
    const size_t n_pts = 15;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + (0.1 + 0.02 * d) * i;  // Chosen by hand.
      }
    }
    return block_logical_coordinates(domain, points);
  }
  ();
  Slab slab(0.0, 1.0);
  TimeId temporal_id(true, 0, Time(slab, Rational(11, 15)));

  runner.simple_action<
      mock_interpolator<metavars>,
      intrp::Actions::ReceivePoints<metavars::InterpolationTargetA>>(
      0, temporal_id, block_logical_coords);

  const auto& holders =
      db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(box);
  const auto& holder =
      get<intrp::Vars::HolderTag<metavars::InterpolationTargetA, metavars>>(
          holders);

  // Should now be a single info in holder, indexed by temporal_id.
  CHECK(holder.infos.size() == 1);
  const auto& vars_info = holder.infos.at(temporal_id);

  // We haven't done any interpolation because we never received
  // volume data from any Elements, so these fields should be empty.
  CHECK(holder.temporal_ids_when_data_has_been_interpolated.empty());
  CHECK(vars_info.interpolation_is_done_for_these_elements.empty());
  CHECK(vars_info.global_offsets.empty());
  CHECK(vars_info.vars.empty());

  // But block_coord_holders should be filled.
  CHECK(vars_info.block_coord_holders == block_logical_coords);

  // There should be no more queued actions; verify this.
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars>>(0));

  // Make sure that the action was not called.
  CHECK(num_calls_of_target_receive_vars == 0);
}
}  // namespace
