// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
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
          typename Metavariables, size_t VolumeDim, typename Frame>
void send_points_to_interpolator(
    db::DataBox<DbTags>& box, Parallel::ConstGlobalCache<Metavariables>& cache,
    const tnsr::I<DataVector, VolumeDim, Frame>& target_points,
    const typename Metavariables::temporal_id::type& temporal_id) noexcept {
  const auto& domain = db::get<::Tags::Domain<VolumeDim, Frame>>(box);
  auto coords = block_logical_coordinates(domain, target_points);

  db::mutate<
      Tags::IndicesOfFilledInterpPoints,
      ::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
      make_not_null(&box),
      [&coords](
          const gsl::not_null<db::item_type<Tags::IndicesOfFilledInterpPoints>*>
              indices_of_filled,
          const gsl::not_null<db::item_type<::Tags::Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>>*>
              vars_dest) noexcept {
        // Because we are sending new points to the interpolator,
        // we know that none of these points have been interpolated to,
        // so clear the list.
        indices_of_filled->clear();

        // We will be filling vars_dest with interpolated data.
        // Here we make sure it is allocated to the correct size.
        if (vars_dest->number_of_grid_points() != coords.size()) {
          *vars_dest = db::item_type<::Tags::Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>>(
              coords.size());
        }
      });

  auto& receiver_proxy =
      Parallel::get_parallel_component<Interpolator<Metavariables>>(cache);
  Parallel::simple_action<
      Actions::ReceivePoints<InterpolationTargetTag>>(
      receiver_proxy, temporal_id, std::move(coords));
}

}  // namespace intrp
