// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace intrp {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Receives interpolated variables from an `Element` on a subset
///  of the target points.
///
/// If interpolated variables for all target points have been received, then
/// - Checks if functions of time are ready (if there are any). If they are not
///   ready, register a simple action callback with the GlobalCache to this
///   action.
/// - Calls `InterpolationTargetTag::post_interpolation_callbacks`
/// - Removes the finished `temporal_id` from `Tags::TemporalIds<TemporalId>`
///   and adds it to `Tags::CompletedTemporalIds<TemporalId>`
/// - Removes `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`,
///   `Tags::IndicesOfFilledInterpPoints`, and
///   `Tags::IndicesOfInvalidInterpPoints` for the finished `temporal_id`.
///
/// Uses:
/// - DataBox:
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::IndicesOfFilledInterpPoints<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
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
/// and intrp::protocols::InterpolationTargetTag
///
/// \note This action can be used only with InterpolationTargets that are
/// non-sequential.
template <typename InterpolationTargetTag>
struct InterpolationTargetVarsFromElement {
  /// For requirements on Metavariables, see InterpolationTarget
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      const std::vector<Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
          vars_src,
      const std::vector<std::optional<
          IdPair<domain::BlockId, tnsr::I<double, Metavariables::volume_dim,
                                          typename ::Frame::BlockLogical>>>>&
          block_logical_coords,
      const std::vector<std::vector<size_t>>& global_offsets,
      const TemporalId& temporal_id,
      const bool vars_have_already_been_received = false) {
    static_assert(
        not InterpolationTargetTag::compute_target_points::is_sequential::value,
        "Use InterpolationTargetGetVarsFromElement only with non-sequential"
        " compute_target_points");
    std::stringstream ss{};
    const ::Verbosity& verbosity = Parallel::get<intrp::Tags::Verbosity>(cache);
    const bool debug_print = verbosity >= ::Verbosity::Debug;
    const bool verbose_print = verbosity >= ::Verbosity::Verbose;
    if (verbose_print) {
      ss << InterpolationTarget_detail::target_output_prefix<
                InterpolationTargetVarsFromElement, InterpolationTargetTag>(
                temporal_id)
         << ", ";
    }
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
      //   InterpolationTargetVarsFromElement is called more than once
      //   with data for the same interpolation point (this is ok,
      //   and add_received_variables handles this).
      // - The only Elements that have not yet called
      //   InterpolationTargetVarsFromElement for this temporal_id are
      //   those that have data only for duplicated interpolation
      //   points, and the InterpolationTarget has already received
      //   that data from other Elements.
      // In this case, the InterpolationTarget proceeds to do its
      // work because it has all the data it needs. There is now
      // one more condition needed for the scenario that gets
      // us inside this 'if':
      // - The InterpolationTarget has already completed its work at
      //   this temporal_id, and it has cleaned up its data structures
      //   for this temporal_id before all of the remaining calls to
      //   InterpolationTargetVarsFromElement have occurred at this
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

    // Call set_up_interpolation only if it has not been called for this
    // temporal_id.
    // If flag_temporal_ids_for_interpolation returns an empty list, then
    // flag_temporal_ids_for_interpolation has already been called for the
    // same temporal_id (by an invocation of InterpolationTargetVarsFromElement
    // by a different Element) and hence set_up_interpolation has already
    // been called.
    if (not InterpolationTarget_detail::flag_temporal_ids_for_interpolation<
                InterpolationTargetTag>(make_not_null(&box),
                                        std::vector<TemporalId>{{temporal_id}})
                .empty()) {
      InterpolationTarget_detail::set_up_interpolation<InterpolationTargetTag>(
          make_not_null(&box), temporal_id, block_logical_coords);
    }

    if (not vars_have_already_been_received) {
      InterpolationTarget_detail::add_received_variables<
          InterpolationTargetTag>(make_not_null(&box), vars_src, global_offsets,
                                  temporal_id);
    }

    if (InterpolationTarget_detail::have_data_at_all_points<
            InterpolationTargetTag>(box, temporal_id, verbosity)) {
      // Check if functions of time are ready on this component. Since this
      // simple action has already been called, we don't need to resend the
      // data, so we just pass empty vectors for vars_src and
      // block_logical_coords
      if (not domain::functions_of_time_are_ready_simple_action_callback<
              domain::Tags::FunctionsOfTime,
              InterpolationTargetVarsFromElement>(
              cache, array_index,
              std::add_pointer_t<ParallelComponent>{nullptr},
              InterpolationTarget_detail::get_temporal_id_value(temporal_id),
              std::nullopt, std::decay_t<decltype(vars_src)>{},
              std::decay_t<decltype(block_logical_coords)>{}, global_offsets,
              temporal_id, true)) {
        return;
      }
      // All the valid points have been interpolated.
      // We throw away the return value of call_callbacks in this case
      // (it is known to be always true; it can be false only for
      //  sequential interpolations, which is static-asserted against above).
      InterpolationTarget_detail::call_callbacks<InterpolationTargetTag>(
          make_not_null(&box), make_not_null(&cache), temporal_id);
      InterpolationTarget_detail::clean_up_interpolation_target<
          InterpolationTargetTag>(make_not_null(&box), temporal_id);
      if (verbose_print) {
        ss << "calling callbacks and cleaning up target.";
        Parallel::printf("%s\n", ss.str());
      }
    } else if (debug_print) {
      ss << "not enough data. Waiting. See Total/valid/invalid points line.";
      Parallel::printf("%s\n", ss.str());
    }
  }
};
}  // namespace Actions
}  // namespace intrp
