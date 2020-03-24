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

namespace InterpolationTarget_detail {

/// Computes the block logical coordinates of an InterpolationTarget.
///
/// get_block_logical_coords is called by an Action of InterpolationTarget.
///
/// Currently two Actions call get_block_logical_coords:
/// - SendPointsToInterpolator (called by AddTemporalIdsToInterpolationTarget
///                             and by FindApparentHorizon)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags,
          typename Metavariables, typename TemporalId>
auto get_block_logical_coords(const db::DataBox<DbTags>& box,
                              Parallel::ConstGlobalCache<Metavariables>& cache,
                              const TemporalId& temporal_id) noexcept {
  const auto& domain =
      db::get<domain::Tags::Domain<Metavariables::volume_dim>>(box);
  return block_logical_coordinates(
      domain, InterpolationTargetTag::compute_target_points::points(
                  box, cache, temporal_id));
}

/// Initializes InterpolationTarget's variables storage and lists of indices
/// corresponding to the supplied block logical coordinates and `temporal_id`.
///
/// set_up_interpolation is called by an Action of InterpolationTarget.
///
/// Currently two Actions call set_up_interpolation:
/// - SendPointsToInterpolator (called by AddTemporalIdsToInterpolationTarget
///                             and by FindApparentHorizon)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags, size_t VolumeDim,
          typename TemporalId>
void set_up_interpolation(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const TemporalId& temporal_id,
    const std::vector<boost::optional<
        IdPair<domain::BlockId,
               tnsr::I<double, VolumeDim, typename ::Frame::Logical>>>>&
        block_logical_coords) noexcept {
  db::mutate<Tags::IndicesOfFilledInterpPoints<TemporalId>,
             Tags::IndicesOfInvalidInterpPoints<TemporalId>,
             Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(
      box, [&block_logical_coords, &temporal_id ](
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
        // Because we are sending new points to the interpolator,
        // we know that none of these points have been interpolated to,
        // so clear the list.
        indices_of_filled->erase(temporal_id);

        // Set the indices of invalid points.
        indices_of_invalid_points->erase(temporal_id);
        for (size_t i = 0; i < block_logical_coords.size(); ++i) {
          if (not block_logical_coords[i]) {
            (*indices_of_invalid_points)[temporal_id].insert(i);
          }
        }

        // At this point we don't know if vars_dest exists in the map;
        // if it doesn't then we want to default construct it.
        auto& vars_dest = (*vars_dest_all_times)[temporal_id];

        // We will be filling vars_dest with interpolated data.
        // Here we make sure it is allocated to the correct size.
        if (vars_dest.number_of_grid_points() != block_logical_coords.size()) {
          vars_dest = Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>(
              block_logical_coords.size());
        }
      });
}
}  // namespace InterpolationTarget_detail

}  // namespace intrp
