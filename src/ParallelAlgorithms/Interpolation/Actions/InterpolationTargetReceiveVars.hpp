// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/VerifyTemporalIdsAndSendPoints.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace intrp {
namespace Tags {
template <typename TemporalId>
struct CompletedTemporalIds;
template <typename TemporalId>
struct PendingTemporalIds;
template <typename TemporalId>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
template <typename TagsList>
struct Variables;
/// \endcond

namespace intrp {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Receives interpolated variables from an `Interpolator` on a subset
///  of the target points.
///
/// If interpolated variables for all target points have been received, then
/// - Calls `InterpolationTargetTag::post_interpolation_callback`
/// - Tells `Interpolator`s that the interpolation is complete
///  (by calling
///  `Actions::CleanUpInterpolator<InterpolationTargetTag>`)
/// - Removes the first `temporal_id` from `Tags::TemporalIds<TemporalId>`
/// - If there are more `temporal_id`s, begins interpolation at the next
///  `temporal_id` (by calling `InterpolationTargetTag::compute_target_points`)
///
/// Uses:
/// - DataBox:
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::IndicesOfFilledInterpPoints<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///   - `Tags::IndicesOfFilledInterpPoints<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars {
  /// For requirements on Metavariables, see InterpolationTarget
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, typename TemporalId,
      Requires<tmpl::list_contains_v<DbTags, Tags::TemporalIds<TemporalId>>> =
          nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const std::vector<Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
          vars_src,
      const std::vector<std::vector<size_t>>& global_offsets,
      const TemporalId& temporal_id) noexcept {
    // Check if we already have completed interpolation at this
    // temporal_id.
    const auto& completed_ids =
        db::get<Tags::CompletedTemporalIds<TemporalId>>(box);
    // (Search from the end because temporal_id is more likely to be
    // at the end of the list then at the beginning.)
    if (UNLIKELY(std::find(completed_ids.rbegin(), completed_ids.rend(),
                           temporal_id) != completed_ids.rend())) {
      // The code will get into this 'if' statement in the following
      // scenario:
      // - There is at least one interpolation point exactly on the
      //   boundary of two or more Elements, so that
      //   InterpolationTargetReceiveVars is called more than once
      //   with data for the same interpolation point (this is ok,
      //   and add_received_variables handles this).
      // - The only Interpolator elements that have not yet called
      //   InterpolationTargetReceiveVars for this temporal_id are
      //   those that have data only for duplicated interpolation
      //   points, and the InterpolationTarget has already received
      //   that data from other Interpolator elements.
      // In this case, the InterpolationTarget proceeds to do its
      // work because it has all the data it needs. There is now
      // one more condition needed for the scenario that gets
      // us inside this 'if':
      // - The InterpolationTarget has already completed its work at
      //   this temporal_id, and it has cleaned up its data structures
      //   for this temporal_id before all of the remaining calls to
      //   InterpolationTargetReceiveVars have occurred at this
      //   temporal_id, and now we are in one of those remaining
      //   calls.
      //
      // If this scenario occurs, we just return. This is because the
      // InterpolationTarget is done and there is nothing left to do
      // at this temporal_id.  Note that if there were extra work to
      // do at this temporal_id, then CompletedTemporalIds would not
      // have an entry for this temporal_id.
      return;
    }

    InterpolationTarget_detail::add_received_variables<InterpolationTargetTag>(
        make_not_null(&box), vars_src, global_offsets, temporal_id);
    if (InterpolationTarget_detail::have_data_at_all_points<
            InterpolationTargetTag>(box, temporal_id)) {
      // All the valid points have been interpolated.
      if (InterpolationTarget_detail::call_callback<InterpolationTargetTag>(
              make_not_null(&box), make_not_null(&cache), temporal_id)) {
        InterpolationTarget_detail::clean_up_interpolation_target<
            InterpolationTargetTag>(make_not_null(&box), temporal_id);
        auto& interpolator_proxy =
            Parallel::get_parallel_component<Interpolator<Metavariables>>(
                cache);
        Parallel::simple_action<
            Actions::CleanUpInterpolator<InterpolationTargetTag>>(
            interpolator_proxy, temporal_id);

        // If we have a sequential target, and there are further
        // temporal_ids, begin interpolation for the next one.
        if (InterpolationTargetTag::compute_target_points::is_sequential::
                value) {
          const auto& temporal_ids =
              db::get<Tags::TemporalIds<TemporalId>>(box);
          if (not temporal_ids.empty()) {
            auto& my_proxy = Parallel::get_parallel_component<
                InterpolationTarget<Metavariables, InterpolationTargetTag>>(
                cache);
            Parallel::simple_action<
                SendPointsToInterpolator<InterpolationTargetTag>>(
                my_proxy, temporal_ids.front());
          } else if (not db::get<Tags::PendingTemporalIds<TemporalId>>(box)
                             .empty()) {
            auto& my_proxy = Parallel::get_parallel_component<
                InterpolationTarget<Metavariables, InterpolationTargetTag>>(
                cache);
            Parallel::simple_action<Actions::VerifyTemporalIdsAndSendPoints<
                InterpolationTargetTag>>(my_proxy);
          }
        }
      }
    }
  }
};
}  // namespace Actions
}  // namespace intrp
