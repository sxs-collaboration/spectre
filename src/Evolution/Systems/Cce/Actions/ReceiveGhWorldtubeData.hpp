// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Stores the boundary data from the GH evolution in the
 * `Cce::InterfaceManagers::GhInterfaceManager`, and sends to the
 * `EvolutionComponent` (template argument) if the data fulfills a prior
 * request.
 *
 * \details If the new data fulfills a prior request submitted to the
 * `Cce::InterfaceManagers::GhInterfaceManager`, this will dispatch the result
 * to `Cce::Actions::SendToEvolution<GhWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` for sending the processed boundary data to
 * the `EvolutionComponent`.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::GhInterfaceManager`
 */
template <typename EvolutionComponent>
struct ReceiveGhWorldtubeData {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<tmpl::list<DbTags...>,
                                           Tags::GhInterfaceManager>> = nullptr>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const TimeStepId& time,
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
      const tnsr::aa<DataVector, 3>& dt_spacetime_metric =
          tnsr::aa<DataVector, 3>{},
      const tnsr::iaa<DataVector, 3>& dt_phi = tnsr::iaa<DataVector, 3>{},
      const tnsr::aa<DataVector, 3>& dt_pi =
          tnsr::aa<DataVector, 3>{}) noexcept {
    db::mutate<Tags::GhInterfaceManager>(
        make_not_null(&box),
        [&spacetime_metric, &phi, &pi, &dt_spacetime_metric, &dt_phi, &dt_pi,
         &time, &cache](const gsl::not_null<
                        std::unique_ptr<InterfaceManagers::GhInterfaceManager>*>
                            interface_manager) noexcept {
          (*interface_manager)
              ->insert_gh_data(time, spacetime_metric, phi, pi,
                               dt_spacetime_metric, dt_phi, dt_pi);
          const auto gh_data =
              (*interface_manager)->retrieve_and_remove_first_ready_gh_data();
          if (static_cast<bool>(gh_data)) {
            Parallel::simple_action<Actions::SendToEvolution<
                GhWorldtubeBoundary<Metavariables>, EvolutionComponent>>(
                Parallel::get_parallel_component<
                    GhWorldtubeBoundary<Metavariables>>(cache),
                get<0>(*gh_data), get<1>(*gh_data));
          }
        });
  }
};
}  // namespace Actions
}  // namespace Cce
