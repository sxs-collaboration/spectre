// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sets up points on an `InterpolationTarget` at a new time
/// and sends these points to an `Interpolator`.
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
///   - `Tags::InterpolatedVars<InterpolationTargetTag>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct SendPointsToInterpolator {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTags, Tags::Times>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double time) noexcept {
    auto coords = InterpolationTarget_detail::block_logical_coords<
        InterpolationTargetTag>(box, cache, time);
    InterpolationTarget_detail::set_up_interpolation<InterpolationTargetTag>(
        make_not_null(&box), time, coords);
    auto& receiver_proxy =
        Parallel::get_parallel_component<Interpolator<Metavariables>>(cache);
    Parallel::simple_action<Actions::ReceivePoints<InterpolationTargetTag>>(
        receiver_proxy, time, std::move(coords));
  }
};

}  // namespace Actions
}  // namespace intrp
