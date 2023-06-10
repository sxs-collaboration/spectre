// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/IdPair.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
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
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      tnsr::I<DataVector, Metavariables::volume_dim,
              typename InterpolationTargetTag::compute_target_points::frame>&&
          coords) {
    db::mutate<intrp::Tags::InterpPointInfo<Metavariables>>(
        [&coords](const gsl::not_null<
                  typename intrp::Tags::InterpPointInfo<Metavariables>::type*>
                      point_infos) {
          get<intrp::Vars::PointInfoTag<InterpolationTargetTag,
                                        Metavariables::volume_dim>>(
              *point_infos) = std::move(coords);
        },
        make_not_null(&box));
  }
};
}  // namespace Actions
}  // namespace intrp
