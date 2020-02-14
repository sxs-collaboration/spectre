// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Cce {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Obtains the CCE boundary data at the specified `time`, and reports it
 * to the `EvolutionComponent` via `Actions::ReceiveWorldtubeData`.
 *
 * \details This uses the `WorldtubeDataManager` to perform all of the work of
 * managing the file buffer, interpolating to the desired time point, and
 * compute the Bondi quantities on the boundary. Once readied, it sends each
 * tensor from the the full `Variables<typename
 * Metavariables::cce_boundary_communication_tags>` back to the
 * `EvolutionComponent`
 *
 * Uses:
 * - DataBox:
 *  - `Tags::H5WorldtubeBoundaryDataManager`
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::Variables<typename
 * Metavariables::cce_boundary_communication_tags>` (every tensor)
 */
template <typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    if (not db::get<Tags::H5WorldtubeBoundaryDataManager>(box)
                .populate_hypersurface_boundary_data(
                    make_not_null(&box), time.substep_time().value())) {
      ERROR("Insufficient boundary data to proceed, exiting early at time " +
            std::to_string(time.substep_time().value()));
    }
    Parallel::receive_data<Cce::ReceiveTags::BoundaryData<
        typename Metavariables::cce_boundary_communication_tags>>(
        Parallel::get_parallel_component<EvolutionComponent>(cache), time,
        db::get<::Tags::Variables<
            typename Metavariables::cce_boundary_communication_tags>>(box),
        true);
  }
};
}  // namespace Actions
}  // namespace Cce
