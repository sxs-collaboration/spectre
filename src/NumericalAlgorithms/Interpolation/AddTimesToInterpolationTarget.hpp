// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/VerifyTimesAndSendPoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Adds times on which this InterpolationTarget
/// should be triggered.
///
/// Invoked on an InterpolationTarget.
///
/// In more detail, does the following:
/// - Adds the passed-in times to PendingTimes.
/// - If the InterpolationTarget is sequential
///   - If Times and PendingTimes were both initially empty,
///     invokes Actions::VerifyTimesAndSendPoints. (Otherwise there
///     (is an interpolation in progress and nothing needs to be done.)
/// - If the InterpolationTarget is not sequential
///   - Invokes Actions::SendPointsToInterpolator on all Times
///   - Invokes Actions::VerifyTimesAndSendPoints if PendingTimes is non-empty.
///
/// Uses:
/// - DataBox:
///   - `Tags::Times`
///   - `Tags::CompletedTimes`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::Times`
template <typename InterpolationTargetTag>
struct AddTimesToInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTags, Tags::Times>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    std::vector<double>&& times) noexcept {
    if constexpr (InterpolationTargetTag::compute_target_points::is_sequential::
                      value) {
      // InterpolationTarget is sequential.
      // - If Tags::Times is non-empty, then there is an
      //   interpolation in progress, so do nothing here.  (If there's
      //   an interpolation in progress, then a later interpolation
      //   will be started as soon as the earlier one finishes (in
      //   InterpolationTargetReceiveVars)).
      // - If not pending_times_was_empty_on_entry, then
      //   there are pending times waiting inside a
      //   VerifyTimesAndSendPoints callback, so do nothing here.
      //   (A later interpolation will be started on the callback).
      // - If Tags::PendingTimes is empty, then we didn't actually
      //   add any pending times above, so do nothing here.
      // - Otherwise, there is no interpolation in progress and there
      //   is no pending_times waiting. So initiate waiting and
      //   interpolation on the pending_times.

      const bool pending_times_was_empty_on_entry =
          db::get<Tags::PendingTimes>(box).empty();

      InterpolationTarget_detail::flag_times_as_pending<InterpolationTargetTag>(
          make_not_null(&box), times);

      if (db::get<Tags::Times>(box).empty() and
          pending_times_was_empty_on_entry and
          not db::get<Tags::PendingTimes>(box).empty()) {
        auto& my_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        Parallel::simple_action<
            Actions::VerifyTimesAndSendPoints<InterpolationTargetTag>>(
            my_proxy);
      }
    } else {
      // InterpolationTarget is not sequential. So everything in
      // Tags::Times should have had interpolation started on it
      // already.  So begin interpolation on every new pending
      // time.

      const std::vector<double> new_pending_times =
          InterpolationTarget_detail::flag_times_as_pending<
              InterpolationTargetTag>(make_not_null(&box), times);

      if (not new_pending_times.empty()) {
        auto& my_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        Parallel::simple_action<
            Actions::VerifyTimesAndSendPoints<InterpolationTargetTag>>(
            my_proxy);
      }
    }
  }
};
}  // namespace Actions
}  // namespace intrp
