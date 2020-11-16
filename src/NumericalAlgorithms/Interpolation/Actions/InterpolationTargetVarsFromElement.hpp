// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
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
/// - Calls `InterpolationTargetTag::post_interpolation_callback`
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
template <typename InterpolationTargetTag>
struct InterpolationTargetVarsFromElement {
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
    static_assert(
        not InterpolationTargetTag::compute_target_points::is_sequential::value,
        "Use InterpolationTargetGetVarsFromElement only with non-sequential"
        " compute_target_points");

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
          make_not_null(&box), temporal_id,
          InterpolationTarget_detail::block_logical_coords<
              InterpolationTargetTag>(box, tmpl::type_<Metavariables>{}));
    }

    InterpolationTarget_detail::add_received_variables<InterpolationTargetTag>(
        make_not_null(&box), vars_src, global_offsets, temporal_id);

    if (InterpolationTarget_detail::have_data_at_all_points<
            InterpolationTargetTag>(box, temporal_id)) {
      // All the valid points have been interpolated.
      // We throw away the return value of call_callback in this case
      // (it is known to be always true; it can be false only for
      //  sequential interpolations, which is static-asserted against above).
      InterpolationTarget_detail::call_callback<InterpolationTargetTag>(
          make_not_null(&box), make_not_null(&cache), temporal_id);
      InterpolationTarget_detail::clean_up_interpolation_target<
          InterpolationTargetTag>(make_not_null(&box), temporal_id);
    }
  }
};
}  // namespace Actions
}  // namespace intrp
