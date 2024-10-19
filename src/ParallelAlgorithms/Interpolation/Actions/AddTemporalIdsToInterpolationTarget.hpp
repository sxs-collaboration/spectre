// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/VerifyTemporalIdsAndSendPoints.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp::Actions {
/// \ingroup ActionsGroup
/// \brief Adds a `temporal_id` on which this InterpolationTarget
/// should be triggered.
///
/// Invoked on an InterpolationTarget.
///
/// In more detail, does the following:
/// - Adds the passed-in `temporal_id` to PendingTemporalIds.
/// - If all the following conditions are met:
///    - CurrentTemporalId doesn't have a value
///    - PendingTemporalIds is not empty,
///    - The temporal id is a LinkedMessageId or PendingTemporalIds was empty
///      upon entry
///   call Actions::VerifyTemporalIdsAndSendPoints. (Otherwise there
///   is an interpolation in progress and nothing needs to be done.)
///
/// Uses:
/// - DataBox:
///   - `Tags::CurrentTemporalId<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::PendingTemporalIds<TemporalId>`
template <typename InterpolationTargetTag>
struct AddTemporalIdsToInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const TemporalId& temporal_id) {
    static_assert(
        InterpolationTargetTag::compute_target_points::is_sequential::value,
        "Actions::AddTemporalIdsToInterpolationTarget can be used only with "
        "sequential targets.");

    const bool pending_temporal_ids_was_empty_on_entry =
        db::get<Tags::PendingTemporalIds<TemporalId>>(box).empty();
    InterpolationTarget_detail::flag_temporal_id_as_pending<
        InterpolationTargetTag>(make_not_null(&box), temporal_id);

    // - If Tags::CurrentTemporalIds has a value, then there is an
    //   interpolation in progress, so do nothing here.  (If there's
    //   an interpolation in progress, then a later interpolation
    //   will be started as soon as the earlier one finishes (in
    //   InterpolationTargetReceiveVars)).
    // - If Tags::PendingTemporalIds is empty, then we didn't actually
    //   add any pending temporal_ids above, so do nothing here.
    // - After we check if TemporalIds is empty and PendingIds isn't, if this is
    //   not a LinkedMessageId, then our indication for sending points was
    //   whether the pending ids was empty at the beginning of this action. But
    //   if this is a LinkedMessageId, it doesn't matter if pending was empty on
    //   entry because of receiving messages out of order. We have to check the
    //   pending id regardless
    if (not db::get<Tags::CurrentTemporalId<TemporalId>>(box).has_value() and
        not db::get<Tags::PendingTemporalIds<TemporalId>>(box).empty() and
        (std::is_same_v<LinkedMessageId<double>, TemporalId> or
         pending_temporal_ids_was_empty_on_entry)) {
      if (Parallel::get<intrp::Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf(
            "%s, Verifying temporal id %s.\n",
            InterpolationTarget_detail::target_output_prefix<
                AddTemporalIdsToInterpolationTarget, InterpolationTargetTag>(),
            temporal_id);
      }

      // Call directly
      Actions::VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>::
          template apply<ParallelComponent>(box, cache, array_index);

    } else if (Parallel::get<intrp::Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      std::stringstream ss{};
      ss << InterpolationTarget_detail::target_output_prefix<
                AddTemporalIdsToInterpolationTarget, InterpolationTargetTag>()
         << ", ";
      using ::operator<<;
      if (db::get<Tags::CurrentTemporalId<TemporalId>>(box).has_value()) {
        ss << "Current temporal id has a value. "
           << db::get<Tags::CurrentTemporalId<TemporalId>>(box).value();
      } else if (db::get<Tags::PendingTemporalIds<TemporalId>>(box).empty()) {
        ss << "Pending temporal ids is empty after insertion.";
      } else {
        if constexpr (std::is_same_v<LinkedMessageId<double>, TemporalId>) {
          ss << "Id is LinkedMessageId<double>.";
        } else {
          ss << "Pending temporal ids was empty on entry";
        }
      }

      Parallel::printf("%s\n", ss.str());
    }
  }
};
}  // namespace intrp::Actions
