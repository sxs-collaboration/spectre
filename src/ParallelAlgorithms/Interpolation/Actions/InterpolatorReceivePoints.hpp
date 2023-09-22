// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain {
class BlockId;
}  // namespace domain
template <typename IdType, typename DataType>
class IdPair;
namespace intrp {
namespace Tags {
struct NumberOfElements;
template <typename Metavariables>
struct InterpolatedVarsHolders;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Receives target points from an InterpolationTarget.
///
/// After receiving the points, interpolates volume data onto them
/// if it already has all the volume data.
///
/// The `iteration` parameter is used to order receives of
/// `block_logical_coords`. Because of the asynchronous nature of communication,
/// it is possible that a more recent set of points arrives before an older set.
/// It is assumed that if a more recent set arrives, then the old set is no
/// longer needed. This `iteration` parameter tags each communication as "more
/// recent" or "older" so if we receive an older set of points after a more
/// recent set, we don't overwrite the more recent set.
///
/// \note If the interpolator receives points with the same iteration, an ERROR
/// will occur.
///
/// Uses:
/// - Databox:
///   - `Tags::NumberOfElements`
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
///   - `Tags::VolumeVarsInfo<Metavariables>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct ReceivePoints {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, size_t VolumeDim>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const typename InterpolationTargetTag::temporal_id::type& temporal_id,
      std::vector<std::optional<
          IdPair<domain::BlockId,
                 tnsr::I<double, VolumeDim, typename ::Frame::BlockLogical>>>>&&
          block_logical_coords,
      const size_t iteration = 0_st) {
    db::mutate<intrp::Tags::InterpolatedVarsHolders<Metavariables>>(
        [&temporal_id, &block_logical_coords, &iteration](
            const gsl::not_null<typename intrp::Tags::InterpolatedVarsHolders<
                Metavariables>::type*>
                vars_holders) {
          auto& vars_infos =
              get<intrp::Vars::HolderTag<InterpolationTargetTag,
                                         Metavariables>>(*vars_holders)
                  .infos;

          // Add the new target interpolation points at this temporal_id. There
          // are two conditions that allow us to overwrite the current target
          // points. Either
          //
          //  1. There are no current target points at the temporal_id, OR
          //  2. There are target points already at this temporal_id, but the
          //     iteration of the new target points is greater than the
          //     iteration of the current target points.
          //
          // If we already have target points and the iteration of the new
          // points is less than or equal to the iteration of the current target
          // points, then we ignore the new points. The new points are outdated
          // and we definitely didn't have any of the new target points in our
          // element by the fact that we have already received the next
          // iteration of points.
          //
          // Whenever we overwrite the target points, we also empty the
          // `interpolation_is_done_for_these_elements` (by virtue of a default
          // constructed `intrp::Vars::Info`) so that we always check every
          // element for this new set of target points.
          if (vars_infos.count(temporal_id) == 0 or
              vars_infos.at(temporal_id).iteration < iteration) {
            vars_infos.insert_or_assign(
                temporal_id,
                intrp::Vars::Info<VolumeDim, typename InterpolationTargetTag::
                                                 vars_to_interpolate_to_target>{
                    std::move(block_logical_coords), iteration});
          } else if (vars_infos.at(temporal_id).iteration == iteration) {
            ERROR(
                "Interpolator received target points at iteration "
                << iteration
                << " twice. Only one set of points per iteration is allowed.");
          }
        },
        make_not_null(&box));

    if (Parallel::get<intrp::Tags::Verbosity>(cache) >= ::Verbosity::Debug) {
      Parallel::printf("%s, received points.\n",
                       InterpolationTarget_detail::interpolator_output_prefix<
                           InterpolationTargetTag>(temporal_id));
    }

    try_to_interpolate<InterpolationTargetTag>(
        make_not_null(&box), make_not_null(&cache), temporal_id);
  }
};

}  // namespace Actions
}  // namespace intrp
