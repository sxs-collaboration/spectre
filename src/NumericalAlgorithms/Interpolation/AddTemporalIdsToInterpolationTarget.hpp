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
                    std::vector<typename Metavariables::temporal_id>&&
                        temporal_ids) noexcept {
    const bool begin_interpolation =
        db::get<Tags::TemporalIds<Metavariables>>(box).empty();

    db::mutate<Tags::TemporalIds<Metavariables>>(
        make_not_null(&box), [&temporal_ids](
                                 const gsl::not_null<db::item_type<
                                     Tags::TemporalIds<Metavariables>>*>
                                     ids) noexcept {
          ids->insert(ids->end(), std::make_move_iterator(temporal_ids.begin()),
                      std::make_move_iterator(temporal_ids.end()));
        });

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
} // namespace Actions
} // namespace intrp
