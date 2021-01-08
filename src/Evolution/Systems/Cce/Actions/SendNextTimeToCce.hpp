// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/PointInfoTag.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Cce::Actions {

/// \cond
struct ReceiveNextElementTime;
/// \endcond

/*!
 * \ingroup ActionsGroup
 * \brief If the element contains the first point in the interpolation
 * collection, sends the next (full) `TimeStepId` to the
 * `Metavariables::cce_boundary_component` to inform the boundary
 * local-time-stepping interpolation/extrapolation.
 *
 * \details After checking the domain against the set of points for the
 * interpolator, this sends the next step time if the current step is also a
 * full step.
 *
 * Uses:
 * - DataBox:
 *   - `Tags::TimeStepId`
 *   - `Tags::Next<Tags::TimeStepId>`
 *   - `Tags::TimeStepper<TimeStepper>`
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: nothing
 */
template <typename InterpolationTargetTag>
struct SendNextTimeToCce {
  using const_global_cache_tags = tmpl::list<::Tags::TimeStepper<TimeStepper>>;
  template <typename DbTags, typename Metavariables, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& block_logical_coords =
        get<intrp::Vars::PointInfoTag<InterpolationTargetTag,
                                      Metavariables::volume_dim>>(
            db::get<intrp::Tags::InterpPointInfo<Metavariables>>(box));
    const std::vector<ElementId<Metavariables::volume_dim>> element_ids{
        {array_index}};
    std::vector<std::optional<IdPair<
        domain::BlockId, tnsr::I<double, 3_st, typename ::Frame::Logical>>>>
        first_valid_block_logical_coords;
    for (const auto& coordinate : block_logical_coords) {
      if (coordinate.has_value()) {
        first_valid_block_logical_coords.push_back(coordinate);
        break;
      }
    }
    if (first_valid_block_logical_coords.empty() ) {
      Parallel::printf(
          "Warning: No valid block logical coordinates found, please ensure "
          "that at least one point for the interpolation is within the domain "
          "to use `SendNextTimeToCce`. No 'next' time is sent, but execution "
          "will continue.\n");
    }
    const auto element_coord_holders = element_logical_coordinates(
        element_ids, first_valid_block_logical_coords);

    if (UNLIKELY(db::get<::Tags::TimeStepId>(box).substep() == 0 and
                 element_coord_holders.count(element_ids[0]) != 0)) {
      // find the next step time via the timestepper
      auto next_step_id = db::get<::Tags::Next<::Tags::TimeStepId>>(box);
      while (next_step_id.substep() != 0) {
        next_step_id =
            db::get<::Tags::TimeStepper<TimeStepper>>(box).next_time_id(
                next_step_id, db::get<::Tags::TimeStep>(box));
      }
      auto& receiver_proxy = Parallel::get_parallel_component<
          typename Metavariables::cce_boundary_component>(cache);
      Parallel::simple_action<Actions::ReceiveNextElementTime>(
          receiver_proxy, db::get<::Tags::TimeStepId>(box),
          std::move(next_step_id));
    }
    return std::forward_as_tuple(std::move(box));
  }
};

/*!
 * \ingroup ActionsGroup
 * \brief Stash the `next_time` in the
 * `Cce::InterfaceManagers::GhInterfaceManager` to inform the local
 * time-stepping logic for boundary interpolation/extrapolation.
 *
 * \details If that information completes a set necessary for generating a
 * requested step's data, then this also dispatches the data to
 * `Cce::Actions::SendToEvolution` boundary computation.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::GhInterfaceManager`
 */
struct ReceiveNextElementTime {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const TimeStepId& time,
                    const TimeStepId& next_time) noexcept {
    if constexpr (tmpl::list_contains_v<tmpl::list<DbTags...>,
                                        Tags::GhInterfaceManager>) {
      db::mutate<Tags::GhInterfaceManager>(
          make_not_null(&box),
          [&cache, &time,
           &next_time](const gsl::not_null<
                       std::unique_ptr<InterfaceManagers::GhInterfaceManager>*>
                           interface_manager) noexcept {
            (*interface_manager)->insert_next_gh_time(time, next_time);
            // if this information permits the evaluation of the next time, then
            // immediately do the evaluation
            const auto gh_data =
                (*interface_manager)->retrieve_and_remove_first_ready_gh_data();
            if (static_cast<bool>(gh_data)) {
              Parallel::simple_action<Actions::SendToEvolution<
                  GhWorldtubeBoundary<Metavariables>,
                  CharacteristicEvolution<Metavariables>>>(
                  Parallel::get_parallel_component<
                      GhWorldtubeBoundary<Metavariables>>(cache),
                  get<0>(*gh_data), get<1>(*gh_data));
            }
          });
    } else {
      ERROR(
          "Tags::GhInterfaceManager must be present in the DataBox to execute "
          "simple action `ReceiveNextElementTime`.");
    }
  }
};
}  // namespace Cce::Actions
