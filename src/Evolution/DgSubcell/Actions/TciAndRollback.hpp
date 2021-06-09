// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/History.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Run the troubled-cell indicator on the candidate solution and perform
 * the time step rollback if needed.
 *
 * The troubled-cell indicator (TCI) is given by the mutator `TciMutator` and
 * can mutate tags in the DataBox, but should do so cautiously. The main reason
 * that this is a mutator is because primitive variables, such as the pressure,
 * are used to check if the solution is physical. In the relativistic case, even
 * just whether or not the primitive variables can be recovered is used as a
 * condition. Note that the evolved variables are projected to the subcells
 * _before_ the TCI is called, and so `Tags::Inactive<variables_tag>` can be
 * used to retrieve the candidate solution on the subcell (e.g. for an RDMP
 * TCI).
 *
 * After rollback, the subcell scheme must project the DG boundary corrections
 * \f$G\f$ to the subcells for the scheme to be conservative. The subcell
 * actions know if a rollback was done because the local mortar data would
 * already be computed.
 */
template <typename TciMutator>
struct TciAndRollback {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t Dim = Metavariables::volume_dim>
  static std::tuple<db::DataBox<DbTags>&&, bool, size_t> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        tmpl::count_if<
            ActionList,
            std::is_same<tmpl::_1, tmpl::pin<TciAndRollback>>>::value == 1,
        "Must have the TciAndRollback action exactly once in the action list "
        "of a phase.");
    static_assert(
        tmpl::count_if<
            ActionList,
            std::is_same<tmpl::_1,
                         tmpl::pin<::Actions::Label<
                             evolution::dg::subcell::Actions::Labels::
                                 BeginSubcellAfterDgRollback>>>>::value == 1,
        "Must have the BeginSubcellAfterDgRollback label exactly once in the "
        "action list of a phase.");

    using variables_tag = typename Metavariables::system::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

    const ActiveGrid active_grid = db::get<Tags::ActiveGrid>(box);
    ASSERT(active_grid == ActiveGrid::Dg,
           "Must be using DG when calling TciAndRollback action.");

    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);

    // Project candidate solution to subcells
    db::mutate<Tags::Inactive<variables_tag>>(
        make_not_null(&box),
        [&dg_mesh, &subcell_mesh](const auto inactive_vars_ptr,
                                  const auto& active_vars) noexcept {
          // Note: strictly speaking, to be conservative this should project uJ
          // instead of u.
          fd::project(inactive_vars_ptr, active_vars, dg_mesh,
                      subcell_mesh.extents());
        },
        db::get<variables_tag>(box));

    // Run RDMP TCI since no user info beyond the input file options are needed
    // for that.
    std::pair self_id{
        Direction<Metavariables::volume_dim>::lower_xi(),
        ElementId<Metavariables::volume_dim>::external_boundary_id()};
    ASSERT(
        db::get<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(box).count(
            self_id) != 0,
        "The self ID is not in the NeighborData.");
    const NeighborData& self_neighbor_data =
        db::get<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(box).at(
            self_id);
    const SubcellOptions& subcell_options = db::get<Tags::SubcellOptions>(box);
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
    // user-specified TCI, since that could be stricter.
    if (not cell_is_troubled) {
      // The reason we pass in the persson_exponent explicitly instead of
      // leaving it to the user is because the value of the exponent that should
      // be used to decide if it is safe to switch back to DG should be
      // `persson_exponent+1` to prevent the code from rapidly switching back
      // and forth between DG and subcell. Rather than trying to enforce this by
      // documentation, the switching back to DG TCI gets passed in the exponent
      // it should use, and to keep the interface between the TCIs consistent,
      // we also pass the exponent in separately here.
      cell_is_troubled = db::mutate_apply<TciMutator>(
          make_not_null(&box), subcell_options.persson_exponent());
    }
    if (cell_is_troubled or
        (subcell_options.always_use_subcells() and
         db::get<::domain::Tags::Element<Metavariables::volume_dim>>(box)
             .external_boundaries()
             .empty())) {
      db::mutate<variables_tag, Tags::Inactive<variables_tag>,
                 ::Tags::HistoryEvolvedVariables<variables_tag>,
                 Tags::ActiveGrid, Tags::DidRollback>(
          make_not_null(&box),
          [&dg_mesh, &subcell_mesh](
              const auto active_vars_ptr, const auto inactive_vars_ptr,
              const auto active_history_ptr,
              const gsl::not_null<ActiveGrid*> active_grid_ptr,
              const gsl::not_null<bool*> did_rollback_ptr) noexcept {
            ASSERT(
                active_history_ptr->size() > 0,
                "We cannot have an empty history when unwinding, that's just "
                "nutty. Did you call the action too early in the action list?");
            // Rollback u^{n+1}* to u^n (undoing the candidate solution) by
            // using the time stepper history.
            *active_vars_ptr = active_history_ptr->most_recent_value();
            fd::project(inactive_vars_ptr, *active_vars_ptr, dg_mesh,
                        subcell_mesh.extents());
            using std::swap;
            swap(*active_vars_ptr, *inactive_vars_ptr);

            // Project the time stepper history to the subcells, excluding the
            // most recent inadmissible history.
            TimeSteppers::History<typename variables_tag::type,
                                  typename dt_variables_tag::type>
                subcell_history{active_history_ptr->integration_order()};
            const auto end_it = std::prev(active_history_ptr->end());
            for (auto it = active_history_ptr->begin(); it != end_it; ++it) {
              subcell_history.insert(it.time_step_id(),
                                     fd::project(it.derivative(), dg_mesh,
                                                 subcell_mesh.extents()));
            }
            *active_history_ptr = std::move(subcell_history);
            *active_grid_ptr = ActiveGrid::Subcell;
            *did_rollback_ptr = true;
            // Note: We do _not_ project the boundary history here because that
            // needs to be done at the lifting stage of the subcell method,
            // since we need to lift G+D instead of the ingredients that go into
            // G+D, which is what we would be projecting here.
          });

      if (UNLIKELY(db::get<::Tags::TimeStepId>(box).slab_number() < 0)) {
        // If we are doing self start, then we need to project the initial guess
        // to the subcells as well.
        //
        // Warning: this unfortunately needs to be kept in sync with the
        //          self-start procedure.
        //
        // Note: if we switch to the subcells then we might have an inconsistent
        //       state between the primitive and conservative variables on the
        //       subcells. The most correct thing is to re-compute the primitive
        //       variables on the subcells, since projecting the conservative
        //       variables is conservative.
        if constexpr (Metavariables::system::
                          has_primitive_and_conservative_vars) {
          db::mutate<
              SelfStart::Tags::InitialValue<variables_tag>,
              SelfStart::Tags::InitialValue<
                  typename Metavariables::system::primitive_variables_tag>>(
              make_not_null(&box),
              [&dg_mesh, &subcell_mesh](
                  const auto initial_vars_ptr,
                  const auto initial_prim_vars_ptr) noexcept {
                // Note: for strict conservation, we need to project uJ instead
                // of just u.
                std::get<0>(*initial_vars_ptr) =
                    fd::project(std::get<0>(*initial_vars_ptr), dg_mesh,
                                subcell_mesh.extents());
                std::get<0>(*initial_prim_vars_ptr) =
                    fd::project(std::get<0>(*initial_prim_vars_ptr), dg_mesh,
                                subcell_mesh.extents());
              });
        } else {
          db::mutate<SelfStart::Tags::InitialValue<variables_tag>>(
              make_not_null(&box),
              [&dg_mesh, &subcell_mesh](const auto initial_vars_ptr) noexcept {
                // Note: for strict conservation, we need to project uJ instead
                // of just u.
                std::get<0>(*initial_vars_ptr) =
                    fd::project(std::get<0>(*initial_vars_ptr), dg_mesh,
                                subcell_mesh.extents());
              });
        }
      }

      return {std::move(box), false,
              tmpl::index_of<
                  ActionList,
                  ::Actions::Label<evolution::dg::subcell::Actions::Labels::
                                       BeginSubcellAfterDgRollback>>::value +
                  1};
    }
    // The unlimited DG solver has passed, so we can remove the current neighbor
    // data.
    db::mutate<subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(
        make_not_null(&box), [](const auto neighbor_data_ptr) noexcept {
          neighbor_data_ptr->clear();
        });

    return {std::move(box), false,
            tmpl::index_of<ActionList, TciAndRollback>::value + 1};
  }
};
}  // namespace evolution::dg::subcell::Actions
