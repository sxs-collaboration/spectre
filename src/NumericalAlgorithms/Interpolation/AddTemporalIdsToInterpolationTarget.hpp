// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
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
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    std::vector<TemporalId>&& temporal_ids) noexcept {
    const bool temporal_id_added_for_the_first_time =
        db::get<Tags::TemporalIds<TemporalId>>(box).empty();

    const std::vector<TemporalId> new_temporal_ids =
        InterpolationTarget_detail::flag_temporal_ids_for_interpolation<
            InterpolationTargetTag>(make_not_null(&box), temporal_ids);

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
            Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
            my_proxy, ids.front());
      }
    } else {
      // InterpolationTarget is not sequential. So begin interpolation
      // on every new temporal_id that has just been added.
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (const auto& id : new_temporal_ids) {
        Parallel::simple_action<
            Actions::SendPointsToInterpolator<InterpolationTargetTag>>(my_proxy,
                                                                       id);
      }
    }
  }
};
}  // namespace Actions
}  // namespace intrp
