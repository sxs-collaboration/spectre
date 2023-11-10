// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace intrp {
template <typename Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget;
namespace Actions {
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sets up points on an `InterpolationTarget` at a new `temporal_id`
/// and sends these points to an `Interpolator`.
///
/// The `iteration` parameter tags each set of points so the `Interpolator`
/// knows which are newer points and which are older points.
///
/// \see `intrp::Actions::ReceivePoints`
///
/// Uses:
/// - DataBox:
///   - `domain::Tags::Domain<3>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `Tags::IndicesOfInvalidInterpPoints`
///   - `Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct SendPointsToInterpolator {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TemporalId& temporal_id,
                    const size_t iteration = 0_st) {
    auto coords = InterpolationTarget_detail::block_logical_coords<
        InterpolationTargetTag>(box, cache, temporal_id);
    InterpolationTarget_detail::set_up_interpolation<InterpolationTargetTag>(
        make_not_null(&box), temporal_id, coords);

    // If all target points are invalid, we need to notify the target as no
    // interpolation is done.
    const auto& invalid_points =
        db::get<Tags::IndicesOfInvalidInterpPoints<TemporalId>>(box);
    if (invalid_points.count(temporal_id) > 0) {
      if (coords.size() == invalid_points.at(temporal_id).size()) {
        auto& receiver_proxy = Parallel::get_parallel_component<
            InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
        // just send empty vectors for the data and global offsets.
        std::vector<Variables<
            typename InterpolationTargetTag::vars_to_interpolate_to_target>>
            vars{};
        std::vector<std::vector<size_t>> global_offsets{};
        Parallel::simple_action<
            Actions::InterpolationTargetReceiveVars<InterpolationTargetTag>>(
            receiver_proxy, vars, global_offsets, temporal_id);
      }
    }

    auto& receiver_proxy =
        Parallel::get_parallel_component<Interpolator<Metavariables>>(cache);
    Parallel::simple_action<Actions::ReceivePoints<InterpolationTargetTag>>(
        receiver_proxy, temporal_id, std::move(coords), iteration);
  }
};

}  // namespace Actions
}  // namespace intrp
