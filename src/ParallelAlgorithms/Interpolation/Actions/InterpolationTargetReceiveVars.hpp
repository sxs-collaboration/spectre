// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/VerifyTemporalIdsAndSendPoints.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace intrp::Tags {
template <typename TemporalId>
struct CompletedTemporalIds;
template <typename TemporalId>
struct PendingTemporalIds;
template <typename TemporalId>
struct TemporalIds;
}  // namespace intrp::Tags
template <typename TagsList>
struct Variables;
/// \endcond

namespace intrp::Actions {
/// \ingroup ActionsGroup
/// \brief Receives interpolated variables from an `Interpolator` on a subset
///  of the target points.
///
/// If interpolated variables for all target points have been received, then
/// - Checks if functions of time are ready (if there are any). If they are not
///   ready, register a simple action callback with the GlobalCache to this
///   action.
/// - Calls `InterpolationTargetTag::post_interpolation_callbacks`
/// - Tells `Interpolator`s that the interpolation is complete
///  (by calling
///  `Actions::CleanUpInterpolator<InterpolationTargetTag>`)
/// - Removes the current id from `Tags::CurrentTemporalId<TemporalId>`
/// - If there are more `temporal_id`s, begins interpolation at the next
///  `temporal_id` (by calling `Actions::VerifyTemporalIdsAndSendPoints`)
///
/// Uses:
/// - DataBox:
///   - `Tags::CurrentTemporalId<TemporalId>`
///   - `Tags::IndicesOfFilledInterpPoints<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::CurrentTemporalId<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///   - `Tags::IndicesOfFilledInterpPoints<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars {
  static_assert(
      InterpolationTargetTag::compute_target_points::is_sequential::value,
      "Actions::InterpolationTargetReceiveVars can be used only with "
      "sequential targets.");
  /// For requirements on Metavariables, see InterpolationTarget
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      const std::vector<Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
          vars_src,
      const std::vector<std::vector<size_t>>& global_offsets,
      const TemporalId& temporal_id,
      const bool vars_have_already_been_received = false) {
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

    if (not vars_have_already_been_received) {
      InterpolationTarget_detail::add_received_variables<
          InterpolationTargetTag>(make_not_null(&box), vars_src, global_offsets,
                                  temporal_id);
    }
    if (InterpolationTarget_detail::have_data_at_all_points<
            InterpolationTargetTag>(box, temporal_id)) {
      // All the valid points have been interpolated.

      // Check if functions of time are ready on this component. Since this
      // simple action has already been called, we don't need to resend the
      // data, so we just pass empty vectors for vars_src and global_offsets
      if (not domain::functions_of_time_are_ready_simple_action_callback<
              domain::Tags::FunctionsOfTime, InterpolationTargetReceiveVars>(
              cache, array_index,
              std::add_pointer_t<ParallelComponent>{nullptr},
              InterpolationTarget_detail::get_temporal_id_value(temporal_id),
              std::nullopt, std::decay_t<decltype(vars_src)>{},
              std::decay_t<decltype(global_offsets)>{}, temporal_id, true)) {
        return;
      }

      if (InterpolationTarget_detail::call_callbacks<InterpolationTargetTag>(
              make_not_null(&box), make_not_null(&cache), temporal_id)) {
        InterpolationTarget_detail::clean_up_interpolation_target<
            InterpolationTargetTag>(make_not_null(&box), temporal_id);
        auto& interpolator_proxy =
            Parallel::get_parallel_component<Interpolator<Metavariables>>(
                cache);
        Parallel::simple_action<
            Actions::CleanUpInterpolator<InterpolationTargetTag>>(
            interpolator_proxy, temporal_id);

        // If there are further pending_temporal_ids, begin interpolation for
        // the next one.
        const auto& current_id =
            db::get<Tags::CurrentTemporalId<TemporalId>>(box);
        using ::operator<<;
        ASSERT(not current_id.has_value(),
               "After a serial interpolation (horizon find) is finished, the "
               "current temporal id shouldn't have a value, but it does "
                   << current_id.value());
        if (not db::get<Tags::PendingTemporalIds<TemporalId>>(box).empty()) {
          // Call directly
          Actions::VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>::
              template apply<ParallelComponent>(box, cache, array_index);
        }
      }
    }
  }
};
}  // namespace intrp::Actions
