// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/IdPair.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "NumericalAlgorithms/Interpolation/PointInfoTag.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Receives interpolation points from an InterpolationTarget.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `intrp::Tags::InterpPointInfo<Metavariables>`
template <typename InterpolationTargetTag>
struct ElementReceiveInterpPoints {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, intrp::Tags::InterpPointInfo<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      std::vector<std::optional<
          IdPair<domain::BlockId, tnsr::I<double, Metavariables::volume_dim,
                                          typename ::Frame::BlockLogical>>>>&&
          block_logical_coords) noexcept {
    db::mutate<intrp::Tags::InterpPointInfo<Metavariables>>(
        make_not_null(&box),
        [&block_logical_coords](
            const gsl::not_null<
                typename intrp::Tags::InterpPointInfo<Metavariables>::type*>
                point_infos) noexcept {
          get<intrp::Vars::PointInfoTag<InterpolationTargetTag,
                                        Metavariables::volume_dim>>(
              *point_infos) = std::move(block_logical_coords);
        });
  }
};
}  // namespace Actions
}  // namespace intrp
