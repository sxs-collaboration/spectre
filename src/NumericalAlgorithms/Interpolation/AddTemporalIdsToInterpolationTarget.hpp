// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/VerifyTemporalIdsAndSendPoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Adds `temporal_id`s on which this InterpolationTarget
/// should be triggered.
///
/// Invoked on an InterpolationTarget.
///
/// In more detail, does the following:
/// - Adds the passed-in `temporal_id`s to PendingTemporalIds.
/// - If the InterpolationTarget is sequential
///   - If TemporalIds and PendingTemporalIds were both initially empty,
///     invokes Actions::VerifyTemporalIdsAndSendPoints. (Otherwise there
///     (is an interpolation in progress and nothing needs to be done.)
/// - If the InterpolationTarget is not sequential
///   - Invokes Actions::SendPointsToInterpolator on all TemporalIds
///   - Invokes Actions::VerifyTemporalIdsAndSendPoints if PendingTemporalIds
///     is non-empty.
///
/// Uses:
/// - DataBox:
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::TemporalIds<TemporalId>`
template <typename InterpolationTargetTag>
struct AddTemporalIdsToInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId,
            Requires<tmpl::list_contains_v<
                DbTags, Tags::TemporalIds<TemporalId>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    std::vector<TemporalId>&& temporal_ids) noexcept {

    if constexpr (InterpolationTargetTag::compute_target_points::is_sequential::
                      value) {
      // InterpolationTarget is sequential.
      // - If Tags::TemporalIds is non-empty, then there is an
      //   interpolation in progress, so do nothing here.  (If there's
      //   an interpolation in progress, then a later interpolation
      //   will be started as soon as the earlier one finishes (in
      //   InterpolationTargetReceiveVars)).
      // - If not pending_temporal_ids_was_empty_on_entry, then
      //   there are pending temporal_ids waiting inside a
      //   VerifyTemporalIdsAndSendPoints callback, so do nothing here.
      //   (A later interpolation will be started on the callback).
      // - If Tags::PendingTemporalIds is empty, then we didn't actually
      //   add any pending temporal_ids above, so do nothing here.
      // - Otherwise, there is no interpolation in progress and there
      //   is no pending_temporal_ids waiting. So initiate waiting and
      //   interpolation on the pending_temporal_ids.

      const bool pending_temporal_ids_was_empty_on_entry =
          db::get<Tags::PendingTemporalIds<TemporalId>>(box).empty();

      InterpolationTarget_detail::flag_temporal_ids_as_pending<
          InterpolationTargetTag>(make_not_null(&box), temporal_ids);

      if (db::get<Tags::TemporalIds<TemporalId>>(box).empty() and
          pending_temporal_ids_was_empty_on_entry and
          not db::get<Tags::PendingTemporalIds<TemporalId>>(box).empty()) {
        auto& my_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        Parallel::simple_action<
            Actions::VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>>(
            my_proxy);
      }
    } else {
      // InterpolationTarget is not sequential. So everything in
      // Tags::TemporalIds should have had interpolation started on it
      // already.  So begin interpolation on every new pending
      // temporal_id.

      const std::vector<TemporalId> new_pending_temporal_ids =
          InterpolationTarget_detail::flag_temporal_ids_as_pending<
              InterpolationTargetTag>(make_not_null(&box), temporal_ids);

      if (not new_pending_temporal_ids.empty()) {
        auto& my_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        Parallel::simple_action<
            Actions::VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>>(
            my_proxy);
      }
    }
  }
};
}  // namespace Actions
}  // namespace intrp
