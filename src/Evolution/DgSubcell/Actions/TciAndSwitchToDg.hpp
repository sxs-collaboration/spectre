// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/History.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Run the troubled-cell indicator on the subcell solution to see if it
 * is safe to switch back to DG.
 *
 * In terms of the DG-subcell/FD hybrid solver, this action is run after the FD
 * step has calculated the solution at \f$t^{n+1}\f$. At this point we check if
 * the FD solution at the new time \f$t^{n+1}\f$ is representable on the DG
 * grid.
 *
 * The algorithm proceeds as follows:
 * 1. If we are using a substep time integrator and are not at the end of a
 *    step, or we are in the self-starting stage of a multistep method, or the
 *    `subcell_options.always_use_subcells() == true`, then we do not run any
 *    TCI or try to go back to DG. We need to avoid reconstructing (in the sense
 *    of the inverse of projecting the DG solution to the subcells) the time
 *    stepper history if there are shocks present in the history, and for
 *    substep methods this is most easily handled by only switching back at the
 *    end of a full time step. During the self-start phase of the multistep time
 *    integrators we integrate over the same region of time at increasingly
 *    higher order, which means if we were on subcell "previously" (since we use
 *    a forward-in-time self-start method the time history is actually in the
 *    future of the current step) then we will very likely need to again switch
 *    to subcell.
 * 2. Reconstruct the subcell solution to the DG grid.
 * 3. Run the relaxed discrete maximum principle (RDMP) troubled-cell indicator
 *    (TCI), checking both the subcell solution at \f$t^{n+1}\f$ and the
 *    reconstructed DG solution at \f$t^{n+1}\f$ to make sure they are
 *    admissible.
 * 4. If the RDMP TCI marked the DG solution as admissible, run the
 *    user-specified mutator TCI `TciMutator`.
 * 5. If the cell is not troubled, and the time integrator type is substep or
 *    the TCI history indicates the entire history for the multistep method is
 *    free of discontinuities, then we can switch back to DG. Switching back to
 *    DG requires swapping the active and inactive evolved variables,
 *    reconstructing the time stepper history, marking the active grid as
 *    `ActiveGrid::Dg`, and clearing the subcell neighbor data.
 * 6. If we are not using a substep method, then record the TCI decision in the
 *    `subcell::Tags::TciGridHistory`.
 *
 * \note Unlike `Actions::TciAndRollback`, this action does _not_ jump back to
 * `Labels::BeginDg`. This is because users may add actions after a time step
 * has been completed. In that sense, it may be more proper to actually check
 * the TCI and switch back to DG at the start of the step rather than the end.
 *
 * \note This action always sets `subcell::Tags::DidRollback` to `false` at the
 * very beginning since this action is called after an FD step has completed.
 *
 * GlobalCache:
 * - Uses:
 *   - `subcell::Tags::SubcellOptions`
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `Tags::TimeStepId`
 *   - `subcell::Tags::ActiveGrid`
 *   - `subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>`
 *   - `subcell::Tags::TciGridHistory`
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::Inactive<System::variables_tag>`
 *   - `System::variables_tag` if the cell is not troubled
 *   - `Tags::HistoryEvolvedVariables` if the cell is not troubled
 *   - `subcell::Tags::ActiveGrid` if the cell is not troubled
 *   - `subcell::Tags::DidRollback` sets to `false`
 *   - `subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>`
 *     if the cell is not troubled
 *   - `subcell::Tags::TciGridHistory` if the time stepper is a multistep method
 */
template <typename TciMutator>
struct TciAndSwitchToDg {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t Dim = Metavariables::volume_dim>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        tmpl::count_if<
            ActionList,
            std::is_same<tmpl::_1, tmpl::pin<TciAndSwitchToDg>>>::value == 1,
        "Must have the TciAndSwitchToDg action exactly once in the action list "
        "of a phase.");

    db::mutate<subcell::Tags::DidRollback>(
        make_not_null(&box),
        [](const gsl::not_null<bool*> did_rollback) noexcept {
          *did_rollback = false;
        });

    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const SubcellOptions& subcell_options = db::get<Tags::SubcellOptions>(box);
    if (time_step_id.substep() != 0 or
        UNLIKELY(time_step_id.slab_number() < 0) or
        UNLIKELY(subcell_options.always_use_subcells())) {
      // The first condition is that for substep time integrators we only allow
      // switching back to DG on step boundaries. This is the easiest way to
      // avoid having a shock in the time stepper history, since there is no
      // history at step boundaries.
      //
      // The second condition is that if we are in the self-start procedure of
      // the time stepper, and we don't want to switch from subcell back to DG
      // during self-start since we integrate over the same temporal region at
      // increasingly higher order.
      //
      // The third condition is that the user has requested we always do
      // subcell, so effectively a finite difference/volume code.
      return {std::move(box)};
    }

    using variables_tag = typename Metavariables::system::variables_tag;

    ASSERT(db::get<Tags::ActiveGrid>(box) == ActiveGrid::Subcell,
           "Must be using subcells when calling TciAndSwitchToDg action.");
    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<subcell::Tags::Mesh<Dim>>(box);

    db::mutate<Tags::Inactive<variables_tag>>(
        make_not_null(&box),
        [&dg_mesh, &subcell_mesh](const auto inactive_vars_ptr,
                                  const auto& active_vars) noexcept {
          // Note: strictly speaking, to be conservative this should reconstruct
          // uJ instead of u.
          fd::reconstruct(inactive_vars_ptr, active_vars, dg_mesh,
                          subcell_mesh.extents());
        },
        db::get<variables_tag>(box));

    // Run RDMP TCI since no user info beyond the input file options are needed
    // for that.
    const std::pair self_id{Direction<Dim>::lower_xi(),
                            ElementId<Dim>::external_boundary_id()};
    ASSERT(
        db::get<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(box).count(
            self_id) != 0,
        "The self ID is not in the NeighborData but should have been added "
        "before TciAndSwitchToDg was called.");
    const NeighborData& self_neighbor_data =
        db::get<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(box).at(
            self_id);

    // Note: we assume the max/min over all neighbors and ourselves at the past
    // time step has been collected into
    // `self_neighbor_data.max/min_variables_values`
    bool cell_is_troubled =
        rdmp_tci(db::get<variables_tag>(box),
                 db::get<Tags::Inactive<variables_tag>>(box),
                 self_neighbor_data.max_variables_values,
                 self_neighbor_data.min_variables_values,
                 subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon());

    // If the RDMP TCI marked the candidate as acceptable, check with the
    // user-specified TCI, since that could be stricter. We pass in the Persson
    // exponent with one added in order to avoid flip-flopping between DG and
    // subcell. That is, `persson_exponent+1.0` is stricter than
    // `persson_exponent`.
    if (not cell_is_troubled) {
      cell_is_troubled = db::mutate_apply<TciMutator>(
          make_not_null(&box), subcell_options.persson_exponent() + 1.0);
    }

    // If the cell is not troubled, then we _might_ be able to switch back to
    // DG. This depends on the type of time stepper we are using:
    // - ADER: Not yet implemented, but here the TCI history is irrelevant
    //         because it is a one-step scheme, so we can always switch.
    // - Multistep: the TCI must have marked the entire evolved variable history
    //              as free of shocks. In practice for LMMs this means the TCI
    //              history is as long as the evolved variables history and that
    //              the entire TCI history is `ActiveGrid::Dg`.
    // - Substep: the easiest is to restrict switching back to DG to step
    //            boundaries where there is no history.
    const auto& time_stepper = db::get<::Tags::TimeStepper<>>(box);
    const bool is_substep_method = time_stepper.number_of_substeps() != 1;
    ASSERT(time_stepper.number_of_substeps() != 0,
           "Don't know how to handle a time stepper with zero substeps. This "
           "might be totally fine, but the case should be thought about.");
    if (const auto& tci_history = db::get<subcell::Tags::TciGridHistory>(box);
        not cell_is_troubled and
        (is_substep_method or
         (tci_history.size() == time_stepper.order() and
          alg::all_of(tci_history, [](const ActiveGrid tci_decision) noexcept {
            return tci_decision == ActiveGrid::Dg;
          })))) {
      db::mutate<variables_tag, Tags::Inactive<variables_tag>,
                 ::Tags::HistoryEvolvedVariables<variables_tag>,
                 Tags::ActiveGrid,
                 subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>,
                 evolution::dg::subcell::Tags::TciGridHistory>(
          make_not_null(&box),
          [&dg_mesh, &subcell_mesh](
              const auto active_vars_ptr, const auto inactive_vars_ptr,
              const auto active_history_ptr,
              const gsl::not_null<ActiveGrid*> active_grid_ptr,
              const auto subcell_neighbor_data_ptr,
              const gsl::not_null<
                  std::deque<evolution::dg::subcell::ActiveGrid>*>
                  tci_grid_history_ptr) noexcept {
            using std::swap;
            swap(*active_vars_ptr, *inactive_vars_ptr);

            // Reconstruct the DG solution for each time in the time stepper
            // history
            using dt_variables_tag =
                db::add_tag_prefix<::Tags::dt, variables_tag>;
            TimeSteppers::History<typename variables_tag::type,
                                  typename dt_variables_tag::type>
                dg_history{active_history_ptr->integration_order()};
            const auto end_it = active_history_ptr->end();
            for (auto it = active_history_ptr->begin(); it != end_it; ++it) {
              dg_history.insert(it.time_step_id(),
                                fd::reconstruct(it.derivative(), dg_mesh,
                                                subcell_mesh.extents()));
            }
            dg_history.most_recent_value() =
                fd::reconstruct(active_history_ptr->most_recent_value(),
                                dg_mesh, subcell_mesh.extents()),
            *active_history_ptr = std::move(dg_history);
            *active_grid_ptr = ActiveGrid::Dg;

            // Clear the neighbor data needed for subcell reconstruction since
            // we have now completed the time step.
            subcell_neighbor_data_ptr->clear();

            // Clear the TCI grid history since we don't need to use it when on
            // the DG grid.
            tci_grid_history_ptr->clear();
          });
      return {std::move(box)};
    }

    if (not is_substep_method) {
      // For multistep methods we need to record the TCI decision history.
      // We track the TCI decision, not which grid we are on because for
      // multistep methods we need the discontinuity to clear the entire
      // history before we can switch back to DG.
      db::mutate<evolution::dg::subcell::Tags::TciGridHistory>(
          make_not_null(&box),
          [cell_is_troubled,
           &time_stepper](const gsl::not_null<
                          std::deque<evolution::dg::subcell::ActiveGrid>*>
                              tci_grid_history) noexcept {
            tci_grid_history->push_front(cell_is_troubled ? ActiveGrid::Subcell
                                                          : ActiveGrid::Dg);
            if (tci_grid_history->size() > time_stepper.order()) {
              tci_grid_history->pop_back();
            }
          });
    }
    return {std::move(box)};
  }
};
}  // namespace evolution::dg::subcell::Actions
