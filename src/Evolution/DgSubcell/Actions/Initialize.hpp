// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Initialize the subcell grid, and switch from DG to subcell if the DG
 * solution is inadmissible.
 *
 * Interior cells are marked as troubled if
 * `subcell_options.always_use_subcells()` is `true` or if
 * `Metavariables::SubcellOptions::DgInitialDataTci::apply` reports that the
 * initial data is not well represented on the DG grid for that cell. Exterior
 * cells are marked as troubled only if
 * `Metavariables::SubcellOptions::subcell_enabled_at_external_boundary` is
 * `true`.
 *
 * If the cell is troubled then `Tags::ActiveGrid` is set to
 * `subcell::ActiveGrid::Subcell` and the `System::variables_tag` become the
 * variables on the subcell grid. The variables are set by projecting from the
 * DG grid for numeric initial data, or by calling
 * `evolution::Initialization::Actions::SetVariables` for analytic initial data.
 * For systems with primitive and conservative variables, only the primitive
 * variables are set by this action, so the conservative variables must be
 * updated in a subsequent action.
 * The `db::add_tag_prefix<Tags::dt, System::variables_tag>` are resized to the
 * subcell grid with an `ASSERT` requiring that they were previously set to the
 * size of the DG grid (this is to reduce the likelihood of them being resized
 * back to the DG grid later).
 *
 * \details `Metavariables::SubcellOptions::DgInitialDataTci::apply` is called
 * with the evolved variables on the DG grid, the projected evolved variables,
 * the DG mesh, the initial RDMP parameters \f$\delta_0\f$ and \f$\epsilon\f$,
 * and the Persson TCI parameter \f$\alpha\f$. The apply function must return a
 * `bool` that is `true` if the cell is troubled.
 *
 * GlobalCache:
 * - Uses:
 *   - `subcell::Tags::SubcellOptions`
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `System::variables_tag`
 * - Adds:
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `subcell::Tags::ActiveGrid`
 *   - `subcell::Tags::DidRollback`
 *   - `subcell::Tags::TciGridHistory`
 *   - `subcell::Tags::GhostDataForReconstruction<Dim>`
 *   - `subcell::Tags::TciDecision`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>`
 *   - `subcell::fd::Tags::DetInverseJacobianLogicalToGrid`
 *   - `subcell::Tags::LogicalCoordinates<Dim>`
 *   - `subcell::Tags::ReconstructionOrder<Dim>` (set as `std::nullopt`)
 *   - `subcell::Tags::Coordinates<Dim, Frame::Grid>` (as compute tag)
 *   - `subcell::Tags::Coordinates<Dim, Frame::Inertial>` (as compute tag)
 * - Removes: nothing
 * - Modifies:
 *   - `System::variables_tag` and `System::primitive_variables_tag` if the cell
 *     is troubled
 *   - `Tags::dt<System::variables_tag>` if the cell is troubled
 */
template <size_t Dim, typename System, typename TciMutator>
struct Initialize {
  using const_global_cache_tags = tmpl::list<Tags::SubcellOptions<Dim>>;

  using simple_tags = tmpl::list<
      Tags::ActiveGrid, Tags::DidRollback, Tags::TciGridHistory,
      Tags::GhostDataForReconstruction<Dim>, Tags::TciDecision,
      Tags::NeighborTciDecisions<Dim>, Tags::DataForRdmpTci,
      fd::Tags::InverseJacobianLogicalToGrid<Dim>,
      fd::Tags::DetInverseJacobianLogicalToGrid,
      subcell::Tags::CellCenteredFlux<typename System::flux_variables, Dim>,
      subcell::Tags::ReconstructionOrder<Dim>>;
  using compute_tags =
      tmpl::list<Tags::MeshCompute<Dim>, Tags::LogicalCoordinatesCompute<Dim>,
                 ::domain::Tags::MappedCoordinates<
                     ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                     subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
                     subcell::Tags::Coordinates>,
                 Tags::InertialCoordinatesCompute<
                     ::domain::CoordinateMaps::Tags::CoordinateMap<
                         Dim, Frame::Grid, Frame::Inertial>>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const SubcellOptions& subcell_options =
        db::get<Tags::SubcellOptions<Dim>>(box);
    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<subcell::Tags::Mesh<Dim>>(box);
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);

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

    const bool subcell_allowed_in_element =
        not std::binary_search(subcell_options.only_dg_block_ids().begin(),
                               subcell_options.only_dg_block_ids().end(),
                               element.id().block_id()) and
        not bordering_dg_block;
    const bool cell_is_not_on_external_boundary =
        db::get<::domain::Tags::Element<Dim>>(box)
            .external_boundaries()
            .empty();

    constexpr bool subcell_enabled_at_external_boundary =
        Metavariables::SubcellOptions::subcell_enabled_at_external_boundary;

    bool cell_is_troubled = subcell_options.always_use_subcells() and
                            (cell_is_not_on_external_boundary or
                             subcell_enabled_at_external_boundary) and
                            subcell_allowed_in_element;

    db::mutate<Tags::NeighborTciDecisions<Dim>>(
        [&element](const auto neighbor_decisions_ptr) {
          neighbor_decisions_ptr->clear();
          for (const auto& [direction, neighbors_in_direction] :
               element.neighbors()) {
            for (const auto& neighbor : neighbors_in_direction.ids()) {
              neighbor_decisions_ptr->insert(
                  std::pair{std::pair{direction, neighbor}, 0});
            }
          }
        },
        make_not_null(&box));

    db::mutate_apply<
        tmpl::list<Tags::ActiveGrid, Tags::DidRollback,
                   typename System::variables_tag, subcell::Tags::TciDecision,
                   subcell::Tags::DataForRdmpTci>,
        typename TciMutator::argument_tags>(
        [&cell_is_troubled, &cell_is_not_on_external_boundary, &dg_mesh,
         subcell_allowed_in_element, &subcell_mesh, &subcell_options](
            const gsl::not_null<ActiveGrid*> active_grid_ptr,
            const gsl::not_null<bool*> did_rollback_ptr,
            const auto active_vars_ptr,
            const gsl::not_null<int*> tci_decision_ptr,
            const auto rdmp_data_ptr, const auto&... args_for_tci) {
          // We don't consider setting the initial grid to subcell as rolling
          // back. Since no time step is undone, we just continue on the
          // subcells as a normal solve.
          *did_rollback_ptr = false;

          *active_grid_ptr = ActiveGrid::Dg;

          *tci_decision_ptr = 0;

          // Now check if the DG solution is admissible. We call the TCI even if
          // the cell is at the boundary since the TCI must also set the past
          // RDMP data.
          std::tuple<int, RdmpTciData> tci_result = TciMutator::apply(
              *active_vars_ptr, subcell_options.initial_data_rdmp_delta0(),
              subcell_options.initial_data_rdmp_epsilon(),
              subcell_options.initial_data_persson_exponent(), args_for_tci...);
          *rdmp_data_ptr = std::move(std::get<1>(std::move(tci_result)));
          *tci_decision_ptr = std::get<0>(tci_result);
          const bool tci_flagged =
              *tci_decision_ptr != 0 and subcell_allowed_in_element;

          if ((cell_is_not_on_external_boundary or
               subcell_enabled_at_external_boundary) and
              (cell_is_troubled or tci_flagged)) {
            cell_is_troubled |= tci_flagged;
            // Swap to subcell grid
            *active_grid_ptr = ActiveGrid::Subcell;
            *active_vars_ptr =
                fd::project(*active_vars_ptr, dg_mesh, subcell_mesh.extents());
          }
        },
        make_not_null(&box));
    if (cell_is_troubled) {
      // Set initial data on subcells. We can avoid projecting the initial data
      // from the DG grid if we have analytic data available. That is the case
      // if either the analytic data / solution type is known at compile-time,
      // or if an analytic data / solution type was selected as initial data at
      // runtime.
      const bool has_analytic_initial_data = [&box]() {
        if constexpr (db::tag_is_retrievable_v<::Tags::AnalyticSolutionOrData,
                                               db::DataBox<DbTagsList>>) {
          (void)box;
          return true;
        } else if constexpr (db::tag_is_retrievable_v<
                                 evolution::initial_data::Tags::InitialData,
                                 db::DataBox<DbTagsList>>) {
          using initial_data_classes = tmpl::at<
              typename Metavariables::factory_creation::factory_classes,
              evolution::initial_data::InitialData>;
          return call_with_dynamic_type<bool, initial_data_classes>(
              &db::get<evolution::initial_data::Tags::InitialData>(box),
              [](const auto* const initial_data) {
                using initial_data_subclass =
                    std::decay_t<decltype(*initial_data)>;
                return is_analytic_data_v<initial_data_subclass> or
                       is_analytic_solution_v<initial_data_subclass>;
              });
        } else {
          ERROR(
              "Expected either '::Tags::AnalyticSolutionOrData' or "
              "'evolution::initial_data::Tags::InitialData' in the DataBox, "
              "but found neither.");
        }
      }();
      if (has_analytic_initial_data) {
        // Set analytic variables on subcells.
        if constexpr (System::has_primitive_and_conservative_vars) {
          db::mutate<typename System::primitive_variables_tag>(
              [&subcell_mesh](const auto prim_vars_ptr) {
                prim_vars_ptr->initialize(subcell_mesh.number_of_grid_points());
              },
              make_not_null(&box));
        }
        evolution::Initialization::Actions::
            SetVariables<Tags::Coordinates<Dim, Frame::ElementLogical>>::apply(
                box, inboxes, cache, array_index, ActionList{},
                std::add_pointer_t<ParallelComponent>{nullptr});
      } else {
        // Project numeric initial data to subcells. It would be good to
        // interpolate numeric initial data directly to the subcell points
        // instead. A simple way to accomplish this is to start the evolution
        // on subcell grids and switch to DG where we can, instead of the
        // reverse (which we currently do).
        if constexpr (System::has_primitive_and_conservative_vars) {
          db::mutate<typename System::primitive_variables_tag>(
              [&dg_mesh, &subcell_mesh](const auto prim_vars_ptr) {
                *prim_vars_ptr = fd::project(*prim_vars_ptr, dg_mesh,
                                             subcell_mesh.extents());
              },
              make_not_null(&box));
        }
      }
      db::mutate<
          db::add_tag_prefix<::Tags::dt, typename System::variables_tag>>(
          [&dg_mesh, &subcell_mesh](const auto dt_vars_ptr) {
            ASSERT(dt_vars_ptr->number_of_grid_points() ==
                       dg_mesh.number_of_grid_points(),
                   "Subcell is resizing the time derivative variables and "
                   "expected them to be the size of the DG grid ("
                       << dg_mesh.number_of_grid_points() << ") but got "
                       << dt_vars_ptr->number_of_grid_points());
            dt_vars_ptr->initialize(subcell_mesh.number_of_grid_points(), 0.0);
          },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::subcell::Actions
