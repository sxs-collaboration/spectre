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
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/VerifyTimesAndSendPoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
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
struct CompletedTimes;
struct PendingTimes;
struct Times;
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
/// - Removes the first time from `Tags::Times`
/// - If there are more timess, begins interpolation at the next
///   time (by calling `InterpolationTargetTag::compute_target_points`)
///
/// Uses:
/// - DataBox:
///   - `Tags::Times`
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `Tags::InterpolatedVars<InterpolationTargetTag>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::Times`
///   - `Tags::CompletedTimes`
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `Tags::InterpolatedVars<InterpolationTargetTag>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars {
  /// For requirements on Metavariables, see InterpolationTarget
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTags, Tags::Times>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const std::vector<Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
          vars_src,
      const std::vector<std::vector<size_t>>& global_offsets,
      const double time) noexcept {
    // Check if we already have completed interpolation at this time.
    const auto& completed_times = db::get<Tags::CompletedTimes>(box);
    // (Search from the end because time is more likely to be
    // at the end of the list then at the beginning.)
    if (UNLIKELY(std::find(completed_times.rbegin(), completed_times.rend(),
                           time) != completed_times.rend())) {
      // The code will get into this 'if' statement in the following
      // scenario:
      // - There is at least one interpolation point exactly on the
      //   boundary of two or more Elements, so that
      //   InterpolationTargetReceiveVars is called more than once
      //   with data for the same interpolation point (this is ok,
      //   and add_received_variables handles this).
      // - The only Interpolator elements that have not yet called
      //   InterpolationTargetReceiveVars for this time are
      //   those that have data only for duplicated interpolation
      //   points, and the InterpolationTarget has already received
      //   that data from other Interpolator elements.
      // In this case, the InterpolationTarget proceeds to do its
      // work because it has all the data it needs. There is now
      // one more condition needed for the scenario that gets
      // us inside this 'if':
      // - The InterpolationTarget has already completed its work at
      //   this time, and it has cleaned up its data structures
      //   for this time before all of the remaining calls to
      //   InterpolationTargetReceiveVars have occurred at this
      //   time, and now we are in one of those remaining
      //   calls.
      //
      // If this scenario occurs, we just return. This is because the
      // InterpolationTarget is done and there is nothing left to do
      // at this time.  Note that if there were extra work to
      // do at this time, then CompletedTimes would not
      // have an entry for this time.
      return;
    }

    InterpolationTarget_detail::add_received_variables<InterpolationTargetTag>(
        make_not_null(&box), vars_src, global_offsets, time);
    if (InterpolationTarget_detail::have_data_at_all_points<
            InterpolationTargetTag>(box, time)) {
      // All the valid points have been interpolated.
      if (InterpolationTarget_detail::call_callback<InterpolationTargetTag>(
              make_not_null(&box), make_not_null(&cache), time)) {
        InterpolationTarget_detail::clean_up_interpolation_target<
            InterpolationTargetTag>(make_not_null(&box), time);
        auto& interpolator_proxy =
            Parallel::get_parallel_component<Interpolator<Metavariables>>(
                cache);
        Parallel::simple_action<
            Actions::CleanUpInterpolator<InterpolationTargetTag>>(
            interpolator_proxy, time);

        // If we have a sequential target, and there are further
        // times, begin interpolation for the next one.
        if (InterpolationTargetTag::compute_target_points::is_sequential::
                value) {
          const auto& times = db::get<Tags::Times>(box);
          if (not times.empty()) {
            auto& my_proxy = Parallel::get_parallel_component<
                InterpolationTarget<Metavariables, InterpolationTargetTag>>(
                cache);
            Parallel::simple_action<
                SendPointsToInterpolator<InterpolationTargetTag>>(
                my_proxy, times.front());
          } else if (not db::get<Tags::PendingTimes>(box).empty()) {
            auto& my_proxy = Parallel::get_parallel_component<
                InterpolationTarget<Metavariables, InterpolationTargetTag>>(
                cache);
            Parallel::simple_action<
                Actions::VerifyTimesAndSendPoints<InterpolationTargetTag>>(
                my_proxy);
          }
        }
      }
    }
  }
};
}  // namespace Actions
}  // namespace intrp
