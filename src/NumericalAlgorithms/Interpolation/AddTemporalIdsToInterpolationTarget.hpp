// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Adds `temporal_id`s on which this InterpolationTarget
/// should be triggered.
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
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    std::vector<TemporalId>&& temporal_ids) noexcept {
    const bool temporal_id_added_for_the_first_time =
        db::get<Tags::TemporalIds<TemporalId>>(box).empty();

    // Some of the temporal_ids may not be new;
    // i.e. AddTemporalIdsToInterpolationTarget may have already been
    // called for them.  So keep track of the new ones.
    std::vector<TemporalId> new_temporal_ids{};

    db::mutate_apply<tmpl::list<Tags::TemporalIds<TemporalId>>,
                     tmpl::list<Tags::CompletedTemporalIds<TemporalId>>>(
        [&temporal_ids, &new_temporal_ids ](
            const gsl::not_null<std::deque<TemporalId>*> ids,
            const std::deque<TemporalId>& completed_ids) noexcept {
          // We allow this Action to be called multiple times with the
          // same temporal_ids (e.g. from each node of a NodeGroup
          // ParallelComponent such as Interpolator). If multiple calls
          // occur, we care only about the first one, and ignore the others.
          // The first call will often begin interpolation.
          // So if multiple calls occur, it is possible that some of them
          // may arrive late, even after interpolation
          // has been completed on one or more of the temporal_ids (and after
          // that id has already been removed from `ids`).  If this happens,
          // we don't want to add the temporal_ids again. For that
          // reason we keep track of the temporal_ids that we have already
          // completed interpolation on.  So here we do not add any temporal_ids
          // that are already present in `ids` or `completed_ids`.
          for (auto& id : temporal_ids) {
            if (std::find(completed_ids.begin(), completed_ids.end(), id) ==
                    completed_ids.end() and
                std::find(ids->begin(), ids->end(), id) == ids->end()) {
              ids->push_back(id);
              new_temporal_ids.push_back(id);
            }
          }
        },
        make_not_null(&box));

    const auto& ids = db::get<Tags::TemporalIds<TemporalId>>(box);
    if (InterpolationTargetTag::compute_target_points::is_sequential::value) {
      // InterpolationTarget is sequential.
      // Begin a single interpolation if one is not already in progress
      // (i.e. waiting for data), and if there are temporal_ids to
      // interpolate.  If there's an interpolation in progress, then a
      // later interpolation will be started as soon as the earlier one
      // finishes (in InterpolationTargetReceiveVars).
      if (temporal_id_added_for_the_first_time and not ids.empty()) {
        auto& my_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        Parallel::simple_action<
            typename InterpolationTargetTag::compute_target_points>(
            my_proxy, ids.front());
      }
    } else {
      // InterpolationTarget is not sequential. So begin interpolation
      // on every new temporal_id that has just been added.
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (const auto& id : new_temporal_ids) {
        Parallel::simple_action<
            typename InterpolationTargetTag::compute_target_points>(my_proxy,
                                                                    id);
      }
    }
  }
};
}  // namespace Actions
}  // namespace intrp
