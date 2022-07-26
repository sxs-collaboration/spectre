// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementReceiveInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Sends interpolation points to all the Elements.
///
/// This action is for the case in which the points are time-independent
/// in the frame of the InterpolationTarget (which may or may not mean that
/// the points are time-independent in the grid frame).
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        not InterpolationTargetTag::compute_target_points::is_sequential::value,
        "Actions::InterpolationTargetSendTimeIndepPointsToElement can be used "
        "only with non-sequential targets, since a sequential target is "
        "time-dependent by definition.");
    auto coords = InterpolationTargetTag::compute_target_points::points(
        box, tmpl::type_<Metavariables>{});
    auto& receiver_proxy = Parallel::get_parallel_component<
        typename InterpolationTargetTag::template interpolating_component<
            Metavariables>>(cache);
    Parallel::simple_action<ElementReceiveInterpPoints<InterpolationTargetTag>>(
        receiver_proxy, std::move(coords));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace intrp
