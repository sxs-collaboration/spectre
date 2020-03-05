// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"

/// \cond
namespace intrp {
template <typename Metavariables>
struct Interpolator;
namespace Actions {
template <typename InterpolationTargetTag>
struct ReceivePoints;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace intrp {

/// Sends to all the Interpolators a list of all the points that need
/// to be interpolated onto.  Also clears information about data that
/// has already been interpolated, since calling this function triggers
/// a new interpolation.
///
/// Called by InterpolationTargetTag::compute_target_points.  This is not an
/// Action, but rather a helper function that is called by every
/// Action (LineSegment, Strahlkorper, etc.) that specifies a
/// particular set of points.
template <typename InterpolationTargetTag, typename DbTags,
          typename Metavariables, typename TemporalId,
          size_t VolumeDim, typename Frame>
void send_points_to_interpolator(
    db::DataBox<DbTags>& box, Parallel::ConstGlobalCache<Metavariables>& cache,
    const tnsr::I<DataVector, VolumeDim, Frame>& target_points,
    const TemporalId& temporal_id) noexcept {
  const auto& domain = db::get<domain::Tags::Domain<VolumeDim>>(box);
  auto coords = block_logical_coordinates(domain, target_points);

  db::mutate<Tags::IndicesOfFilledInterpPoints<TemporalId>,
             Tags::IndicesOfInvalidInterpPoints<TemporalId>,
             Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(
      make_not_null(&box),
      [&coords, &temporal_id ](
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_filled,
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_invalid_points,
          const gsl::not_null<std::unordered_map<
              TemporalId, Variables<typename InterpolationTargetTag::
                                        vars_to_interpolate_to_target>>*>
              vars_dest_all_times) noexcept {
        // At this point we don't know if vars_dest exists in the map;
        // if it doesn't then we want to default construct it.
        auto& vars_dest = (*vars_dest_all_times)[temporal_id];
        // Because we are sending new points to the interpolator,
        // we know that none of these points have been interpolated to,
        // so clear the list.
        indices_of_filled->erase(temporal_id);

        // Set the indices of invalid points.
        indices_of_invalid_points->erase(temporal_id);
        for (size_t i = 0; i < coords.size(); ++i) {
          if (not coords[i]) {
            (*indices_of_invalid_points)[temporal_id].insert(i);
          }
        }

        // We will be filling vars_dest with interpolated data.
        // Here we make sure it is allocated to the correct size.
        if (vars_dest.number_of_grid_points() != coords.size()) {
          vars_dest = Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>(
              coords.size());
        }
      });

  auto& receiver_proxy =
      Parallel::get_parallel_component<Interpolator<Metavariables>>(cache);
  Parallel::simple_action<Actions::ReceivePoints<InterpolationTargetTag>>(
      receiver_proxy, temporal_id, std::move(coords));
}

}  // namespace intrp
