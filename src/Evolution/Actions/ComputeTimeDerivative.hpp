// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace tuples {
template <typename...>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Compute the time derivative of the system variables
 *
 * The `system::compute_time_derivative` operator is invoked with its
 * `argument_tags`. The result of this computation is stored in
 * `db::add_tag_prefix<step_prefix, variables_tag>`, where `step_prefix` is
 * retrieved from the `Metavariables::temporal_id`. For evolution systems this
 * prefix is generally `Tags::dt`.
 *
 * Uses:
 * - Metavariables:
 *   - `temporal_id::step_prefix`
 * - System:
 *   - `variables_tag`
 * - DataBox:
 *   - db::add_tag_prefix<step_prefix, variables_tag>
 *   - All elements in `compute_time_derivative::argument_tags`
 *
 * DataBox changes:
 * - Modifies:
 *   - `TimeDerivativeComputer::return_tags<step_prefix>`
 */
template <typename TimeDerivativeComputer>
struct ComputeTimeDerivative {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    // Notes:
    // - dt_variables is not zeroed and the operator cannot assume this.
    // - We retrieve the `step_prefix` from the `Metavariables` (as opposed to
    // hard-coding `Tags::dt`) to retain consistency with other actions that do
    // the same (for instance `dg::Actions::ApplyFluxes` that is not specific to
    // evolution systems)
    db::mutate_apply<typename TimeDerivativeComputer::template return_tags<
                         Metavariables::temporal_id::template step_prefix>,
                     typename TimeDerivativeComputer::argument_tags>(
        TimeDerivativeComputer{}, make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
