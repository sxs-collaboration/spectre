// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/ElementReceiveInterpPoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Sends interpolation points to all the Elements.
///
/// This action is for the case in which the points are time-independent.
///
/// This action should be placed in the Registration PDAL for
/// InterpolationTarget.
///
/// Uses:
/// - DataBox:
///   - Anything that the particular
///     InterpolationTargetTag::compute_target_points needs from the DataBox.
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename InterpolationTargetTag>
struct InterpolationTargetSendTimeIndepPointsToElements {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename Metavariables,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto coords = InterpolationTarget_detail::block_logical_coords<
        InterpolationTargetTag>(box, tmpl::type_<Metavariables>{});
    auto& receiver_proxy = Parallel::get_parallel_component<
        typename InterpolationTargetTag::interpolating_component>(cache);
    Parallel::simple_action<ElementReceiveInterpPoints<InterpolationTargetTag>>(
        receiver_proxy, std::move(coords));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace intrp
