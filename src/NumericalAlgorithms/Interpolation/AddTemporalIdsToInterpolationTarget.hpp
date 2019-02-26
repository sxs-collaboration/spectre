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
///   - `Tags::TemporalIds<Metavariables>`
///   - `Tags::CompletedTemporalIds<Metavariables>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::TemporalIds<Metavariables>`
template <typename InterpolationTargetTag>
struct AddTemporalIdsToInterpolationTarget {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, typename Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<typename Metavariables::temporal_id::type>&&
                        temporal_ids) noexcept {
    const bool begin_interpolation =
        db::get<Tags::TemporalIds<Metavariables>>(box).empty();

    db::mutate_apply<tmpl::list<Tags::TemporalIds<Metavariables>>,
                     tmpl::list<Tags::CompletedTemporalIds<Metavariables>>>(
        [&temporal_ids](
            const gsl::not_null<
                db::item_type<Tags::TemporalIds<Metavariables>>*>
                ids,
            const db::item_type<Tags::CompletedTemporalIds<Metavariables>>&
                completed_ids) noexcept {
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
            }
          }
        },
        make_not_null(&box));

    // Begin interpolation if it is not already in progress
    // (i.e. waiting for data), and if there are temporal_ids to
    // interpolate.  If there's an interpolation in progress, then a
    // later interpolation will be started as soon as the earlier one
    // finishes.
    const auto& ids = db::get<Tags::TemporalIds<Metavariables>>(box);
    if (begin_interpolation and not ids.empty()) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          typename InterpolationTargetTag::compute_target_points>(my_proxy,
                                                                  ids.front());
    }
  }
};
}  // namespace Actions
}  // namespace intrp
