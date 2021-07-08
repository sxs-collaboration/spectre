// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
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
 * \details See the template partial specializations of this class for details
 * on the different strategies for each component type.
 */
template <typename BoundaryComponent, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution;

/*!
 * \ingroup ActionsGroup
 * \brief Computes Bondi boundary data from GH evolution variables and sends the
 * result to the `EvolutionComponent` (template argument).
 *
 * \details After the computation, this action will call
 * `Cce::Actions::ReceiveWorldtubeData` on the `EvolutionComponent` with each of
 * the types from `typename Metavariables::cce_boundary_communication_tags` sent
 * as arguments
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::Variables<typename
 *   Metavariables::cce_boundary_communication_tags>` (every tensor)
 */
template <typename BoundaryComponent, typename EvolutionComponent>
struct SendToEvolution;

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
template <typename Metavariables, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution<H5WorldtubeBoundary<Metavariables>,
                                         EvolutionComponent> {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<std::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    auto hdf5_lock = Parallel::local_branch(Parallel::get_parallel_component<
                         observers::ObserverWriter<Metavariables>>(cache))
                         ->template local_synchronous_action<
                             observers::Actions::GetLockPointer<
                                 observers::Tags::H5FileLock>>();
    bool successfully_populated = false;
    db::mutate<Tags::H5WorldtubeBoundaryDataManager,
               ::Tags::Variables<
                   typename Metavariables::cce_boundary_communication_tags>>(
        make_not_null(&box),
        [&successfully_populated, &time, &hdf5_lock](
            const gsl::not_null<std::unique_ptr<Cce::WorldtubeDataManager>*>
                worldtube_data_manager,
            const gsl::not_null<Variables<
                typename Metavariables::cce_boundary_communication_tags>*>
                boundary_variables) noexcept {
          successfully_populated =
              (*worldtube_data_manager)
                  ->populate_hypersurface_boundary_data(
                      boundary_variables, time.substep_time().value(),
                      hdf5_lock);
        });
    if (not successfully_populated) {
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

/*!
 * \ingroup ActionsGroup
 * \brief Calculates the analytic boundary data at the specified `time`, and
 * sends the resulting Bondi-Sachs boundary data to the `EvolutionComponent`
 *
 * \details This uses the `Cce::AnalyticBoundaryDataManager` to
 * perform all of the work of calculating the analytic boundary solution, which
 * in turn uses derived classes of `Cce::Solutions::WorldtubeData` to calculate
 * the metric data before it is transformed to Bondi-Sachs variables.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::AnalyticWordltubeBoundaryDataManager`
 */
template <typename Metavariables, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution<
    AnalyticWorldtubeBoundary<Metavariables>, EvolutionComponent> {
  template <typename ParallelComponent, typename DbTagList, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagList,
                      ::Tags::Variables<typename Metavariables::
                                            cce_boundary_communication_tags>>) {
      bool successfully_populated = false;
      db::mutate<Tags::AnalyticBoundaryDataManager,
                 ::Tags::Variables<
                     typename Metavariables::cce_boundary_communication_tags>>(
          make_not_null(&box),
          [&successfully_populated, &time](
              const gsl::not_null<Cce::AnalyticBoundaryDataManager*>
                  worldtube_data_manager,
              const gsl::not_null<Variables<
                  typename Metavariables::cce_boundary_communication_tags>*>
                  boundary_variables) noexcept {
            successfully_populated =
                (*worldtube_data_manager)
                    .populate_hypersurface_boundary_data(
                        boundary_variables, time.substep_time().value());
          });

      if (not successfully_populated) {
        ERROR("Insufficient boundary data to proceed, exiting early at time "
              << time.substep_time().value());
      }
      Parallel::receive_data<Cce::ReceiveTags::BoundaryData<
          typename Metavariables::cce_boundary_communication_tags>>(
          Parallel::get_parallel_component<EvolutionComponent>(cache), time,
          db::get<::Tags::Variables<
              typename Metavariables::cce_boundary_communication_tags>>(box),
          true);
    } else {
      ERROR(
          "Did not find required tag `::Tags::Variables<typename "
          "Metavariables::cce_boundary_communication_tags>` in the DataBox");
    }
  }
};

/*!
 * \ingroup ActionsGroup
 * \brief Submits a request for CCE boundary data at the specified `time` to the
 * `Cce::InterfaceManagers::GhInterfaceManager`, and sends the data to the
 * `EvolutionComponent` (template argument) if it is ready.
 *
 * \details This uses the `Cce::InterfaceManagers::GhInterfaceManager` to
 * perform all of the work of managing the buffer of data sent from the GH
 * system and interpolating if necessary and supported. This dispatches then to
 * `Cce::Actions::SendToEvolution<GhWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` if the boundary data is ready, otherwise
 * simply submits the request and waits for data to become available via
 * `Cce::Actions::ReceiveGhWorldtubeData`, which will call
 * `Cce::Actions::SendToEvolution<GhWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` as soon as the data becomes available.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::GhInterfaceManager`
 */
template <typename Metavariables, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution<GhWorldtubeBoundary<Metavariables>,
                                         EvolutionComponent> {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<std::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    db::mutate<Tags::GhInterfaceManager>(
        make_not_null(&box),
        [&time, &cache](const gsl::not_null<
                        std::unique_ptr<InterfaceManagers::GhInterfaceManager>*>
                            interface_manager) noexcept {
          (*interface_manager)->request_gh_data(time);
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

/// \cond
template <typename Metavariables, typename EvolutionComponent>
struct SendToEvolution<GhWorldtubeBoundary<Metavariables>, EvolutionComponent> {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<std::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const TimeStepId& time,
                    const InterfaceManagers::GhInterfaceManager::gh_variables&
                        gh_variables) noexcept {
    db::mutate<::Tags::Variables<
        typename Metavariables::cce_boundary_communication_tags>>(
        make_not_null(&box),
        [&gh_variables](
            const gsl::not_null<Variables<
                typename Metavariables::cce_boundary_communication_tags>*>
                boundary_variables,
            const double extraction_radius, const double l_max) noexcept {
          create_bondi_boundary_data(
              boundary_variables,
              get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
                  gh_variables),
              get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(
                  gh_variables),
              get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
                  gh_variables),
              extraction_radius, l_max);
        },
        db::get<InitializationTags::ExtractionRadius>(box),
        db::get<Tags::LMax>(box));
    Parallel::receive_data<Cce::ReceiveTags::BoundaryData<
        typename Metavariables::cce_boundary_communication_tags>>(
        Parallel::get_parallel_component<EvolutionComponent>(cache), time,
        db::get<::Tags::Variables<
            typename Metavariables::cce_boundary_communication_tags>>(box),
        true);
  }
};
/// \endcond

}  // namespace Actions
}  // namespace Cce
