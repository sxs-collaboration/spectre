// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
struct ReceivePoints;
}  // namespace Actions
}  // namespace intrp
template <typename IdType, typename DataType>
class IdPair;
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
struct IndicesOfFilledInterpPoints;
template <typename Metavariables>
struct InterpolatedVarsHolders;
struct NumberOfElements;
}  // namespace Tags
}  // namespace intrp
namespace domain {
class BlockId;
}  // namespace domain
/// \endcond

namespace InterpTargetTestHelpers {

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked = void;  // not needed.
  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags<tmpl::list<
          typename Metavariables::InterpolationTargetA::compute_target_points>>;

  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;
};

template <typename InterpolationTargetTag>
struct MockReceivePoints {
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
      const typename Metavariables::temporal_id::type& temporal_id,
      std::vector<IdPair<domain::BlockId,
                         tnsr::I<double, VolumeDim, typename Frame::Logical>>>&&
          block_coord_holders) noexcept {
    db::mutate<intrp::Tags::InterpolatedVarsHolders<Metavariables>>(
        make_not_null(&box),
        [
          &temporal_id, &block_coord_holders
        ](const gsl::not_null<
            db::item_type<intrp::Tags::InterpolatedVarsHolders<Metavariables>>*>
              vars_holders) noexcept {
          auto& vars_infos =
              get<intrp::Vars::HolderTag<InterpolationTargetTag,
                                         Metavariables>>(*vars_holders)
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

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator::
          template return_tag_list<Metavariables>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using replace_these_simple_actions = tmpl::list<intrp::Actions::ReceivePoints<
      typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions = tmpl::list<
      MockReceivePoints<typename Metavariables::InterpolationTargetA>>;
};

template <typename MetaVariables, typename DomainCreator,
          typename InterpolationTargetOption, typename BlockCoordHolder>
void test_interpolation_target(
    const DomainCreator& domain_creator, InterpolationTargetOption options,
    const BlockCoordHolder& expected_block_coord_holders) noexcept {
  using metavars = MetaVariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      typename MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTarget =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars,
                                    typename metavars::InterpolationTargetA>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars>>;
  tuples::get<MockDistributedObjectsTagTarget>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<mock_interpolation_target<
                   metavars, typename metavars::InterpolationTargetA>>{});
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      mock_interpolator<metavars>>{});

  tuples::TaggedTuple<typename metavars::InterpolationTargetA> tuple_of_opts(
      std::move(options));

  MockRuntimeSystem runner{std::move(tuple_of_opts), std::move(dist_objects)};

  runner.template simple_action<
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>,
      ::intrp::Actions::InitializeInterpolationTarget<
          typename metavars::InterpolationTargetA>>(
      0, domain_creator.create_domain());

  runner.template simple_action<mock_interpolator<metavars>,
                                ::intrp::Actions::InitializeInterpolator>(0);

  const auto& box_target =
      runner
          .template algorithms<mock_interpolation_target<
              metavars, typename metavars::InterpolationTargetA>>()
          .at(0)
          .template get_databox<typename mock_interpolation_target<
              metavars,
              typename metavars::InterpolationTargetA>::initial_databox>();

  const auto& box_interpolator =
      runner.template algorithms<mock_interpolator<metavars>>()
          .at(0)
          .template get_databox<
              typename mock_interpolator<metavars>::initial_databox>();

  Slab slab(0.0, 1.0);
  TimeId temporal_id(true, 0, Time(slab, 0));

  runner.template simple_action<
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>,
      typename metavars::InterpolationTargetA::compute_target_points>(
      0, temporal_id);

  // This should not have changed.
  CHECK(
      db::get<::intrp::Tags::IndicesOfFilledInterpPoints>(box_target).empty());

  // Should be no queued actions in mock_interpolation_target
  CHECK(runner.template is_simple_action_queue_empty<mock_interpolation_target<
            metavars, typename metavars::InterpolationTargetA>>(0));

  // But there should be one in mock_interpolator
  runner.template invoke_queued_simple_action<mock_interpolator<metavars>>(
      0);

  // Should be no more queued actions in mock_interpolator
  CHECK(runner.template is_simple_action_queue_empty<
        mock_interpolator<metavars>>(0));

  const auto& vars_holders =
      db::get<intrp::Tags::InterpolatedVarsHolders<metavars>>(
          box_interpolator);
  const auto& vars_infos =
      get<intrp::Vars::HolderTag<typename metavars::InterpolationTargetA,
                                 metavars>>(vars_holders)
          .infos;
  // Should be one entry in the vars_infos
  CHECK(vars_infos.size() == 1);
  const auto& info = vars_infos.at(temporal_id);
  const auto& block_coord_holders = info.block_coord_holders;

  // Check number of points
  const size_t number_of_points = expected_block_coord_holders.size();
  CHECK(block_coord_holders.size() == number_of_points);

  for (size_t i = 0; i < number_of_points; ++i) {
    CHECK(block_coord_holders[i].id == expected_block_coord_holders[i].id);
    CHECK_ITERABLE_APPROX(block_coord_holders[i].data,
                          expected_block_coord_holders[i].data);
  }

  // Call again at a different temporal_id
  TimeId new_temporal_id(true, 0, Time(slab, 1));
  runner.template simple_action<
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>,
      typename metavars::InterpolationTargetA::compute_target_points>(
      0, new_temporal_id);
  runner.template invoke_queued_simple_action<mock_interpolator<metavars>>(
      0);

  // Should be two entries in the vars_infos
  CHECK(vars_infos.size() == 2);
  const auto& new_block_coord_holders =
      vars_infos.at(new_temporal_id).block_coord_holders;
  for (size_t i = 0; i < number_of_points; ++i) {
    CHECK(new_block_coord_holders[i].id == expected_block_coord_holders[i].id);
    CHECK_ITERABLE_APPROX(new_block_coord_holders[i].data,
                          expected_block_coord_holders[i].data);
  }
}

}  // namespace InterpTargetTestHelpers
