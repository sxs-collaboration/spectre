// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Shell.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetLineSegment.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
struct ReceiveInterpolationPoints;
}  // namespace Actions
}  // namespace intrp
template <typename IdType, typename DataType>
class IdPair;
namespace Parallel {
template <typename Metavariables> class ConstGlobalCache;
} // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
} // namespace db
namespace intrp {
namespace Tags {
struct IndicesOfFilledInterpPoints;
template <typename Metavariables, size_t VolumeDim>
struct InterpolatedVarsHolders;
struct NumberOfElements;
} // namespace Tags
} // namespace intrp
/// \endcond

namespace {

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked = void; // not needed.
  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags<
          tmpl::list<intrp::Actions::LineSegment<InterpolationTargetTag, 3,
                                                 Frame::Inertial>>>;

  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables, 3,
                                                            ::Frame::Inertial>>;
};

template <typename InterpolationTargetTag>
struct MockReceiveInterpolationPoints {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t VolumeDim,
            Requires<tmpl::list_contains_v<
                DbTags, typename ::intrp::Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const typename Metavariables::temporal_id& temporal_id,
      std::vector<IdPair<domain::BlockId,
                         tnsr::I<double, VolumeDim, typename Frame::Logical>>>&&
          block_coord_holders) noexcept {
    db::mutate<intrp::Tags::InterpolatedVarsHolders<Metavariables, VolumeDim>>(
        make_not_null(&box),
        [&temporal_id, &block_coord_holders](
            const gsl::not_null<
                db::item_type<intrp::Tags::InterpolatedVarsHolders<
                    Metavariables, VolumeDim>>*>
                vars_holders) {
          auto& vars_infos =
              get<intrp::Vars::HolderTag<InterpolationTargetTag, Metavariables,
                                         VolumeDim>>(*vars_holders)
                  .infos;

          // Add the target interpolation points at this timestep.
          vars_infos.emplace(std::make_pair(
              temporal_id,
              intrp::Vars::Info<VolumeDim, typename InterpolationTargetTag::
                                               vars_to_interpolate_to_target>{
                  std::move(block_coord_holders)}));
        });
  }
};

template <typename Metavariables, size_t VolumeDim>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator<
          VolumeDim>::template return_tag_list<Metavariables>>;

  using component_being_mocked = intrp::Interpolator<Metavariables, VolumeDim>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::ReceiveInterpolationPoints<
          typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions = tmpl::list<MockReceiveInterpolationPoints<
      typename Metavariables::InterpolationTargetA>>;
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_target_points =
        intrp::Actions::LineSegment<InterpolationTargetA, 3, Frame::Inertial>;
    using type = typename compute_target_points::options_type;
  };
  using temporal_id = Time;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables, 3>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.LineSegment",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTarget =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars, metavars::InterpolationTargetA>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars, 3>>;
  tuples::get<MockDistributedObjectsTagTarget>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<mock_interpolation_target<
                   metavars, metavars::InterpolationTargetA>>{});
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      mock_interpolator<metavars, 3>>{});

  // Options for LineSegment
  intrp::OptionHolders::LineSegment<3> line_segment_opts({{1.0, 1.0, 1.0}},
                                                         {{2.4, 2.4, 2.4}}, 15);
  tuples::TaggedTuple<metavars::InterpolationTargetA> tuple_of_opts(
      line_segment_opts);

  MockRuntimeSystem runner{tuple_of_opts, std::move(dist_objects)};

  const auto domain_creator =
      DomainCreators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::InitializeInterpolationTarget<
          metavars::InterpolationTargetA>>(0, domain_creator.create_domain());

  runner.simple_action<mock_interpolator<metavars, 3>,
                       ::intrp::Actions::InitializeInterpolator<3>>(0);

  const auto& box_target =
      runner
          .template algorithms<mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>>()
          .at(0)
          .template get_databox<typename mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>::initial_databox>();

  const auto& box_interpolator =
      runner.template algorithms<mock_interpolator<metavars, 3>>()
          .at(0)
          .template get_databox<
              typename mock_interpolator<metavars, 3>::initial_databox>();

  Slab slab(0.0, 1.0);
  Time temporal_id(slab, 0);

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::LineSegment<metavars::InterpolationTargetA, 3,
                                    Frame::Inertial>>(0, temporal_id);

  // This should not have changed.
  CHECK(
      db::get<::intrp::Tags::IndicesOfFilledInterpPoints>(box_target).empty());

  // Should be no queued actions in mock_interpolation_target
  CHECK(runner.is_simple_action_queue_empty<
        mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(
      0));

  // But there should be one in mock_interpolator
  runner.invoke_queued_simple_action<mock_interpolator<metavars, 3>>(0);

  // Should be no more queued actions in mock_interpolator
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars, 3>>(0));

  const auto& vars_holders =
      db::get<intrp::Tags::InterpolatedVarsHolders<metavars, 3>>(
          box_interpolator);
  const auto& vars_infos =
      get<intrp::Vars::HolderTag<metavars::InterpolationTargetA, metavars, 3>>(
          vars_holders)
          .infos;
  // Should be one entry in the vars_infos
  CHECK(vars_infos.size() == 1);
  const auto& info = vars_infos.at(temporal_id);
  const auto& block_coord_holders = info.block_coord_holders;

  // Should be 15 points.
  CHECK(block_coord_holders.size() == 15);

  const auto expected_block_coord_holders = [&domain_creator]() {
    const size_t n_pts = 15;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + 0.1 * i;  // Worked out by hand.
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();
  for (size_t i = 0; i < 15; ++i) {
    CHECK(block_coord_holders[i].id == expected_block_coord_holders[i].id);
    CHECK_ITERABLE_APPROX(block_coord_holders[i].data,
                          expected_block_coord_holders[i].data);
  }

  // Call again at a different temporal_id
  Time new_temporal_id(slab, 1);
  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::LineSegment<metavars::InterpolationTargetA, 3,
                                    Frame::Inertial>>(0, new_temporal_id);
  runner.invoke_queued_simple_action<mock_interpolator<metavars, 3>>(0);

  // Should be two entries in the vars_infos
  CHECK(vars_infos.size() == 2);
  const auto& new_block_coord_holders =
      vars_infos.at(new_temporal_id).block_coord_holders;
  for (size_t i = 0; i < 15; ++i) {
    CHECK(new_block_coord_holders[i].id == expected_block_coord_holders[i].id);
    CHECK_ITERABLE_APPROX(new_block_coord_holders[i].data,
                          expected_block_coord_holders[i].data);
  }
}

}  // namespace
