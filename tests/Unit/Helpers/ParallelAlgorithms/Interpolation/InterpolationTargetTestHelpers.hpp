// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
class GlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
template <typename TemporalId>
struct IndicesOfFilledInterpPoints;
template <typename TemporalId>
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
  using const_global_cache_tags = tmpl::flatten<tmpl::append<
      Parallel::get_const_global_cache_tags_from_actions<tmpl::list<
          typename Metavariables::InterpolationTargetA::compute_target_points>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     intrp::Actions::InitializeInterpolationTarget<
                         Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};

template <typename InterpolationTargetTag>
struct MockReceivePoints {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, size_t VolumeDim,
      Requires<tmpl::list_contains_v<DbTags, ::intrp::Tags::NumberOfElements>> =
          nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::InterpolationTargetA::temporal_id::type&
          temporal_id,
      std::vector<std::optional<
          IdPair<domain::BlockId,
                 tnsr::I<double, VolumeDim, typename Frame::BlockLogical>>>>&&
          block_coord_holders) {
    db::mutate<intrp::Tags::InterpolatedVarsHolders<Metavariables>>(
        make_not_null(&box),
        [&temporal_id, &block_coord_holders](
            const gsl::not_null<typename intrp::Tags::InterpolatedVarsHolders<
                Metavariables>::type*>
                vars_holders) {
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
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              Actions::SetupDataBox,
              intrp::Actions::InitializeInterpolator<
                  intrp::Tags::VolumeVarsInfo<
                      Metavariables, typename Metavariables::
                                         InterpolationTargetA::temporal_id>,
                  intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using replace_these_simple_actions = tmpl::list<intrp::Actions::ReceivePoints<
      typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions = tmpl::list<
      MockReceivePoints<typename Metavariables::InterpolationTargetA>>;
};

template <typename MetaVariables, typename InterpolationTargetOptionTag,
          typename DomainCreator, typename BlockCoordHolder>
void test_interpolation_target(
    const DomainCreator& domain_creator,
    typename InterpolationTargetOptionTag::type options,
    const BlockCoordHolder& expected_block_coord_holders) {
  using metavars = MetaVariables;
  using temporal_id_type =
      typename metavars::InterpolationTargetA::temporal_id::type;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;
  using interp_component = mock_interpolator<metavars>;

  tuples::TaggedTuple<InterpolationTargetOptionTag,
                      domain::Tags::Domain<MetaVariables::volume_dim>>
      tuple_of_opts{std::move(options),
                    std::move(domain_creator.create_domain())};
  ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Initialization);
  ActionTesting::emplace_component<interp_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_component<target_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, 0));
  static_assert(
      std::is_same_v<typename metavars::InterpolationTargetA::temporal_id::type,
                     double> or
          std::is_same_v<
              typename metavars::InterpolationTargetA::temporal_id::type,
              TimeStepId>,
      "Unsupported temporal_id type");
  if constexpr (std::is_same_v<temporal_id_type, double>) {
    ActionTesting::simple_action<target_component,
                                 intrp::Actions::SendPointsToInterpolator<
                                     typename metavars::InterpolationTargetA>>(
        make_not_null(&runner), 0, temporal_id.substep_time().value());
  } else if constexpr (std::is_same_v<temporal_id_type, TimeStepId>) {
    ActionTesting::simple_action<target_component,
                                 intrp::Actions::SendPointsToInterpolator<
                                     typename metavars::InterpolationTargetA>>(
        make_not_null(&runner), 0, temporal_id);
  }

  // This should not have changed.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
            runner, 0)
            .empty());

  // Should be no queued actions in mock_interpolation_target
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));

  // But there should be one in mock_interpolator

  ActionTesting::invoke_queued_simple_action<interp_component>(
      make_not_null(&runner), 0);

  // Should be no more queued actions in mock_interpolator
  CHECK(
      ActionTesting::is_simple_action_queue_empty<mock_interpolator<metavars>>(
          runner, 0));

  const auto& vars_holders = ActionTesting::get_databox_tag<
      interp_component, intrp::Tags::InterpolatedVarsHolders<metavars>>(runner,
                                                                        0);
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
    CHECK(block_coord_holders[i].value().id ==
          expected_block_coord_holders[i].value().id);
    CHECK_ITERABLE_APPROX(block_coord_holders[i].value().data,
                          expected_block_coord_holders[i].value().data);
  }

  // Call again at a different temporal_id
  TimeStepId new_temporal_id(true, 0, Time(slab, 1));
  ActionTesting::simple_action<target_component,
                               intrp::Actions::SendPointsToInterpolator<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, new_temporal_id);
  ActionTesting::invoke_queued_simple_action<interp_component>(
      make_not_null(&runner), 0);

  // Should be two entries in the vars_infos
  CHECK(vars_infos.size() == 2);
  const auto& new_block_coord_holders =
      vars_infos.at(new_temporal_id).block_coord_holders;
  for (size_t i = 0; i < number_of_points; ++i) {
    CHECK(new_block_coord_holders[i].value().id ==
          expected_block_coord_holders[i].value().id);
    CHECK_ITERABLE_APPROX(new_block_coord_holders[i].value().data,
                          expected_block_coord_holders[i].value().data);
  }
}

}  // namespace InterpTargetTestHelpers
