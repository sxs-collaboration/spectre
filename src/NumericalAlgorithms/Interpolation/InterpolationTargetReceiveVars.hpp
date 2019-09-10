// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace intrp {
template <class Metavariables>
struct Interpolator;
template <typename Metavariables, typename InterpolationTargetTag>
class InterpolationTarget;
namespace Actions {
template <typename InterpolationTargetTag>
struct CleanUpInterpolator;
}  // namespace Actions
namespace Tags {
struct IndicesOfFilledInterpPoints;
template <typename Metavariables>
struct CompletedTemporalIds;
template <typename Metavariables>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace intrp {

namespace InterpolationTarget_detail {

// apply_callback accomplishes the overload for the
// two signatures of callback functions.
// Uses SFINAE on return type.
template <typename T, typename DbTags, typename Metavariables>
auto apply_callback(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::ConstGlobalCache<Metavariables>*> cache,
    const typename Metavariables::temporal_id::type& temporal_id) noexcept
    -> decltype(T::post_interpolation_callback::apply(box, cache, temporal_id),
                bool()) {
  return T::post_interpolation_callback::apply(box, cache, temporal_id);
}

template <typename T, typename DbTags, typename Metavariables>
auto apply_callback(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::ConstGlobalCache<Metavariables>*> cache,
    const typename Metavariables::temporal_id::type& temporal_id) noexcept
    -> decltype(T::post_interpolation_callback::apply(*box, *cache,
                                                      temporal_id),
                bool()) {
  T::post_interpolation_callback::apply(*box, *cache, temporal_id);
  // For the simpler callback function, we will always clean up volume data, so
  // we return true here.
  return true;
}

/// Calls the callback function, tells interpolators to clean up the current
/// temporal_id, and then if there are more temporal_ids to be interpolated,
/// starts the next one.
template <typename InterpolationTargetTag, typename DbTags,
          typename Metavariables>
void callback_and_cleanup(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::ConstGlobalCache<Metavariables>*>
        cache) noexcept {
  const auto temporal_id =
      db::get<Tags::TemporalIds<Metavariables>>(*box).front();

  // apply_callback should return true if we are done with this
  // temporal_id.  It should return false only if the callback
  // calls another `intrp::Action` that needs the volume data at this
  // same temporal_id.  If it returns false, we exit here and do not
  // clean up.
  const bool done_with_temporal_id =
      apply_callback<InterpolationTargetTag>(box, cache, temporal_id);

  if (not done_with_temporal_id) {
    return;
  }

  // We are now done with this temporal_id, so we can pop it and
  // clean up volume data associated with it.
  db::mutate<Tags::TemporalIds<Metavariables>,
             Tags::CompletedTemporalIds<Metavariables>>(
      box, [](const gsl::not_null<
                  db::item_type<Tags::TemporalIds<Metavariables>>*>
                  ids,
              const gsl::not_null<
                  db::item_type<Tags::CompletedTemporalIds<Metavariables>>*>
                  completed_ids) noexcept {
        completed_ids->push_back(ids->front());
        ids->pop_front();
        // We want to keep track of all completed temporal_ids to deal with
        // the possibility of late calls to
        // AddTemporalIdsToInterpolationTarget.  We could keep all
        // completed_ids forever, but we probably don't want it to get too
        // large, so we limit its size.  We assume that
        // asynchronous calls to AddTemporalIdsToInterpolationTarget do not span
        // more than 10 temporal_ids.
        if(completed_ids->size() > 10) {
          completed_ids->pop_front();
        }
      });

  // Tell interpolators to clean up at this temporal_id for this
  // InterpolationTargetTag.
  auto& interpolator_proxy =
      Parallel::get_parallel_component<Interpolator<Metavariables>>(*cache);
  Parallel::simple_action<Actions::CleanUpInterpolator<InterpolationTargetTag>>(
      interpolator_proxy, temporal_id);

  // If there are further temporal_ids, begin interpolation for
  // the next one.
  const auto& temporal_ids = db::get<Tags::TemporalIds<Metavariables>>(*box);
  if (not temporal_ids.empty()) {
    auto& my_proxy = Parallel::get_parallel_component<
        InterpolationTarget<Metavariables, InterpolationTargetTag>>(*cache);
    Parallel::simple_action<
        typename InterpolationTargetTag::compute_target_points>(
        my_proxy, temporal_ids.front());
  }
}

}  // namespace InterpolationTarget_detail

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
/// - Removes the first `temporal_id` from `Tags::TemporalIds<Metavariables>`
/// - If there are more `temporal_id`s, begins interpolation at the next
///  `temporal_id` (by calling `InterpolationTargetTag::compute_target_points`)
///
/// Uses:
/// - DataBox:
///   - `Tags::TemporalIds<Metavariables>`
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::TemporalIds<Metavariables>`
///   - `Tags::CompletedTemporalIds<Metavariables>`
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars {
  /// For requirements on Metavariables, see InterpolationTarget
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const std::vector<db::item_type<::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>>&
          vars_src,
      const std::vector<std::vector<size_t>>& global_offsets) noexcept {
    db::mutate<
        Tags::IndicesOfFilledInterpPoints,
        ::Tags::Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
        make_not_null(&box),
        [
          &vars_src, &global_offsets
        ](const gsl::not_null<db::item_type<Tags::IndicesOfFilledInterpPoints>*>
              indices_of_filled,
          const gsl::not_null<db::item_type<::Tags::Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>>*>
              vars_dest) noexcept {
          const size_t npts_dest = vars_dest->number_of_grid_points();
          const size_t nvars = vars_dest->number_of_independent_components;
          for (size_t j = 0; j < global_offsets.size(); ++j) {
            const size_t npts_src = global_offsets[j].size();
            for (size_t i = 0; i < npts_src; ++i) {
              // If a point is on the boundary of two (or more)
              // elements, it is possible that we have received data
              // for this point from more than one Interpolator.
              // This will rarely occur, but it does occur, e.g. when
              // a point is exactly on some symmetry
              // boundary (such as the x-y plane) and this symmetry
              // boundary is exactly the boundary between two
              // elements.  If this happens, we accept the first
              // duplicated point, and we ignore subsequent
              // duplicated points.  The points are easy to keep track
              // of because global_offsets uniquely identifies them.
              if (indices_of_filled->insert(global_offsets[j][i]).second) {
                for (size_t v = 0; v < nvars; ++v) {
                  // clang-tidy: no pointer arithmetic
                  vars_dest->data()[global_offsets[j][i] +   // NOLINT
                                    v * npts_dest] =         // NOLINT
                      vars_src[j].data()[i + v * npts_src];  // NOLINT
                }
              }
            }
          }
        });

    if (db::get<Tags::IndicesOfFilledInterpPoints>(box).size() ==
        db::get<::Tags::Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
            box)
            .number_of_grid_points()) {
      // All the points have been interpolated.
      InterpolationTarget_detail::callback_and_cleanup<InterpolationTargetTag>(
          make_not_null(&box), make_not_null(&cache));
    }
  }
};
}  // namespace Actions
}  // namespace intrp
