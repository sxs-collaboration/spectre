// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Tags/NeighborMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/History.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Run the troubled-cell indicator on the candidate solution and perform
 * the time step rollback if needed.
 *
 * Interior cells are marked as troubled if
 * `subcell_options.always_use_subcells()` is `true`, or if either the RDMP
 * troubled-cell indicator (TCI) or the TciMutator reports the cell is
 * troubled. Exterior cells are marked as troubled only if
 * `Metavariables::SubcellOptions::subcell_enabled_at_external_boundary` is
 * `true`.
 *
 * The troubled-cell indicator (TCI) given by the mutator `TciMutator` can
 * mutate tags in the DataBox, but should do so cautiously. The main reason that
 * this is a mutator is because primitive variables, such as the pressure, are
 * used to check if the solution is physical. In the relativistic case, even
 * just whether or not the primitive variables can be recovered is used as a
 * condition. Note that the evolved variables are projected to the subcells
 * _after_ the TCI is called and marks the cell as troubled.
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
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

    const ActiveGrid active_grid = db::get<Tags::ActiveGrid>(box);
    ASSERT(active_grid == ActiveGrid::Dg,
           "Must be using DG when calling TciAndRollback action.");

    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const bool cell_has_external_boundary =
        not element.external_boundaries().empty();

    constexpr bool subcell_enabled_at_external_boundary =
        Metavariables::SubcellOptions::subcell_enabled_at_external_boundary;
    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);

    const SubcellOptions& subcell_options =
        db::get<Tags::SubcellOptions<Dim>>(box);
    bool cell_is_troubled =
        subcell_options.always_use_subcells() or
        (subcell_options.use_halo() and [&box]() -> bool {
          for (const auto& [_, neighbor_decision] :
               db::get<evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
                   box)) {
            if (neighbor_decision != 0) {
              return true;
            }
          }
          return false;
        }());

    // Loop over block neighbors and if neighbor id is inside of
    // subcell_options.only_dg_block_ids(), then bordering DG-only block
    const bool bordering_dg_block = alg::any_of(
        element.neighbors(),
        [&subcell_options](const auto& direction_and_neighbor) {
          const size_t first_block_id =
              direction_and_neighbor.second.ids().begin()->block_id();
          return std::binary_search(subcell_options.only_dg_block_ids().begin(),
                                    subcell_options.only_dg_block_ids().end(),
                                    first_block_id);
        });

    // Subcell is allowed in the element if 2 conditions are met:
    // (i)  The current element block id is not marked as DG only
    // (ii) The current element is not bordering a DG only block.
    const bool subcell_allowed_in_element =
        not std::binary_search(subcell_options.only_dg_block_ids().begin(),
                               subcell_options.only_dg_block_ids().end(),
                               element.id().block_id()) and
        not bordering_dg_block;

    // The reason we pass in the persson_exponent explicitly instead of
    // leaving it to the user is because the value of the exponent that
    // should be used to decide if it is safe to switch back to DG should be
    // `persson_exponent+1` to prevent the code from rapidly switching back
    // and forth between DG and subcell. Rather than trying to enforce this
    // by documentation, the switching back to DG TCI gets passed in the
    // exponent it should use, and to keep the interface between the TCIs
    // consistent, we also pass the exponent in separately here.
    std::tuple<int, RdmpTciData> tci_result = db::mutate_apply<TciMutator>(
        make_not_null(&box), subcell_options.persson_exponent(),
        not subcell_allowed_in_element);

    const int tci_decision = std::get<0>(tci_result);
    db::mutate<Tags::TciDecision>(
        [&tci_decision](const gsl::not_null<int*> tci_decision_ptr) {
          *tci_decision_ptr = tci_decision;
        },
        make_not_null(&box));

    cell_is_troubled |= (tci_decision != 0);

    // If either:
    //
    // 1. we are not allowed to do subcell in this block
    // 2. the element is at an outer boundary _and_ we aren't allow to go to
    //    subcell at an outer boundary.
    // 3. the cell is not troubled
    //
    // then we can remove the current neighbor data and update the RDMP TCI
    // data.
    if (not subcell_allowed_in_element or
        (cell_has_external_boundary and
         not subcell_enabled_at_external_boundary) or
        not cell_is_troubled) {
      db::mutate<subcell::Tags::GhostDataForReconstruction<Dim>,
                 subcell::Tags::DataForRdmpTci>(
          [&tci_result](const auto neighbor_data_ptr,
                        const gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr) {
            neighbor_data_ptr->clear();
            *rdmp_tci_data_ptr = std::move(std::get<1>(std::move(tci_result)));
          },
          make_not_null(&box));
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    db::mutate<variables_tag, ::Tags::HistoryEvolvedVariables<variables_tag>,
               Tags::ActiveGrid, Tags::DidRollback,
               subcell::Tags::GhostDataForReconstruction<Dim>>(
        [&dg_mesh, &element, &subcell_mesh](
            const auto active_vars_ptr, const auto active_history_ptr,
            const gsl::not_null<ActiveGrid*> active_grid_ptr,
            const gsl::not_null<bool*> did_rollback_ptr,
            const gsl::not_null<FixedHashMap<
                maximum_number_of_neighbors(Dim),
                std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
                boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
                ghost_data_ptr,
            const FixedHashMap<
                maximum_number_of_neighbors(Dim),
                std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim>,
                boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
                neighbor_meshes,
            const size_t ghost_zone_size) {
          ASSERT(active_history_ptr->size() > 0,
                 "We cannot have an empty history when unwinding, that's just "
                 "nutty. Did you call the action too early in the action "
                 "list?");
          // Rollback u^{n+1}* to u^n (undoing the candidate solution).
          //
          // Note: strictly speaking, to be conservative this should project
          // uJ instead of u.
          *active_vars_ptr = fd::project(active_history_ptr->latest_value(),
                                         dg_mesh, subcell_mesh.extents());

          // Project the time stepper history to the subcells, excluding the
          // most recent inadmissible history.
          active_history_ptr->undo_latest();
          active_history_ptr->map_entries(
              [&dg_mesh, &subcell_mesh](const auto entry) {
                *entry = fd::project(*entry, dg_mesh, subcell_mesh.extents());
              });
          *active_grid_ptr = ActiveGrid::Subcell;
          *did_rollback_ptr = true;
          // Project the neighbor data we were sent for reconstruction since
          // the neighbor might have sent DG volume data instead of ghost data
          // in order to elide projections when they aren't necessary.
          for (const auto& [directional_element_id, neighbor_mesh] :
               neighbor_meshes) {
            evolution::dg::subcell::insert_or_update_neighbor_volume_data<
                false>(ghost_data_ptr,
                       ghost_data_ptr->at(directional_element_id)
                           .neighbor_ghost_data_for_reconstruction(),
                       0, directional_element_id, neighbor_mesh, element,
                       subcell_mesh, ghost_zone_size);
          }

          // Note: We do _not_ project the boundary history here because
          // that needs to be done at the lifting stage of the subcell
          // method, since we need to lift G+D instead of the ingredients
          // that go into G+D, which is what we would be projecting here.
        },
        make_not_null(&box),
        db::get<evolution::dg::Tags::NeighborMesh<Dim>>(box),
        Metavariables::SubcellOptions::ghost_zone_size(box));

    if (UNLIKELY(db::get<::Tags::TimeStepId>(box).slab_number() < 0)) {
      // If we are doing self start, then we need to project the initial
      // guess to the subcells as well.
      //
      // Warning: this unfortunately needs to be kept in sync with the
      //          self-start procedure.
      //
      // Note: if we switch to the subcells then we might have an
      // inconsistent
      //       state between the primitive and conservative variables on the
      //       subcells. The most correct thing is to re-compute the
      //       primitive variables on the subcells, since projecting the
      //       conservative variables is conservative.
      if constexpr (Metavariables::system::
                        has_primitive_and_conservative_vars) {
        db::mutate<
            SelfStart::Tags::InitialValue<variables_tag>,
            SelfStart::Tags::InitialValue<
                typename Metavariables::system::primitive_variables_tag>>(
            [&dg_mesh, &subcell_mesh](const auto initial_vars_ptr,
                                      const auto initial_prim_vars_ptr) {
              // Note: for strict conservation, we need to project uJ
              // instead of just u.
              std::get<0>(*initial_vars_ptr) =
                  fd::project(std::get<0>(*initial_vars_ptr), dg_mesh,
                              subcell_mesh.extents());
              std::get<0>(*initial_prim_vars_ptr) =
                  fd::project(std::get<0>(*initial_prim_vars_ptr), dg_mesh,
                              subcell_mesh.extents());
            },
            make_not_null(&box));
      } else {
        db::mutate<SelfStart::Tags::InitialValue<variables_tag>>(
            [&dg_mesh, &subcell_mesh](const auto initial_vars_ptr) {
              // Note: for strict conservation, we need to project uJ
              // instead of just u.
              std::get<0>(*initial_vars_ptr) =
                  fd::project(std::get<0>(*initial_vars_ptr), dg_mesh,
                              subcell_mesh.extents());
            },
            make_not_null(&box));
      }
    }

    return {Parallel::AlgorithmExecution::Continue,
            tmpl::index_of<
                ActionList,
                ::Actions::Label<evolution::dg::subcell::Actions::Labels::
                                     BeginSubcellAfterDgRollback>>::value +
                1};
  }
};
}  // namespace evolution::dg::subcell::Actions
