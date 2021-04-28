// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DgSubcell/TciStatus.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
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
 * cells are never marked as troubled, because subcell doesn't yet support
 * boundary conditions.
 *
 * If the cell is troubled then `Tags::ActiveGrid` is set to
 * `subcell::ActiveGrid::Subcell`, the `System::variables_tag` become the
 * variables on the subcell grid set by calling
 * `evolution::Initialization::Actions::SetVariables`, and
 * ` Tags::Inactive<System::variables_tag>` become the (inadmissible) DG
 * solution.
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
 *   - `subcell::Tags::Inactive<System::variables_tag>`
 *   - `subcell::Tags::TciGridHistory`
 *   - `subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>`
 *   - `subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>`
 *   - `subcell::fd::Tags::DetInverseJacobianLogicalToGrid`
 *   - `subcell::Tags::LogicalCoordinates<Dim>`
 *   - `subcell::Tags::Corodinates<Dim, Frame::Grid>` (as compute tag)
 *   - `subcell::Tags::TciStatusCompute<Dim>`
 * - Removes: nothing
 * - Modifies:
 *   - `System::variables_tag` if the cell is troubled
 */
template <size_t Dim, typename System, typename TciMutator>
struct Initialize {
  using const_global_cache_tags = tmpl::list<Tags::SubcellOptions>;

  using simple_tags =
      tmpl::list<Tags::Mesh<Dim>, Tags::ActiveGrid, Tags::DidRollback,
                 Tags::Inactive<typename System::variables_tag>,
                 Tags::TciGridHistory,
                 Tags::NeighborDataForReconstructionAndRdmpTci<Dim>,
                 fd::Tags::InverseJacobianLogicalToGrid<Dim>,
                 fd::Tags::DetInverseJacobianLogicalToGrid>;
  using compute_tags =
      tmpl::list<Tags::LogicalCoordinatesCompute<Dim>,
                 ::domain::Tags::MappedCoordinates<
                     ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                     subcell::Tags::Coordinates<Dim, Frame::Logical>,
                     subcell::Tags::Coordinates>,
                 Tags::TciStatusCompute<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const SubcellOptions& subcell_options = db::get<Tags::SubcellOptions>(box);
    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim> subcell_mesh = fd::mesh(dg_mesh);
    // Note: we currently cannot do subcell at boundaries, so only set on
    // interior elements.
    bool cell_is_troubled = subcell_options.always_use_subcells() and
                            db::get<::domain::Tags::Element<Dim>>(box)
                                .external_boundaries()
                                .empty();
    db::mutate_apply<tmpl::list<subcell::Tags::Mesh<Dim>, Tags::ActiveGrid,
                                Tags::DidRollback,
                                Tags::Inactive<typename System::variables_tag>,
                                typename System::variables_tag>,
                     typename TciMutator::argument_tags>(
        [&cell_is_troubled, &dg_mesh, &subcell_mesh, &subcell_options](
            const gsl::not_null<Mesh<Dim>*> subcell_mesh_ptr,
            const gsl::not_null<ActiveGrid*> active_grid_ptr,
            const gsl::not_null<bool*> did_rollback_ptr,
            const auto inactive_vars_ptr, const auto active_vars_ptr,
            const auto&... args_for_tci) noexcept {
          // We don't consider setting the initial grid to subcell as rolling
          // back. Since no time step is undone, we just continue on the
          // subcells as a normal solve.
          *did_rollback_ptr = false;

          *subcell_mesh_ptr = subcell_mesh;
          *active_grid_ptr = ActiveGrid::Dg;
          fd::project(inactive_vars_ptr, *active_vars_ptr, dg_mesh,
                      subcell_mesh.extents());
          // Now check if the DG solution is admissible
          cell_is_troubled |= TciMutator::apply(
              *active_vars_ptr, *inactive_vars_ptr,
              subcell_options.initial_data_rdmp_delta0(),
              subcell_options.initial_data_rdmp_epsilon(),
              subcell_options.initial_data_persson_exponent(), args_for_tci...);
          if (cell_is_troubled) {
            // Swap grid
            *active_grid_ptr = ActiveGrid::Subcell;
            using std::swap;
            swap(*active_vars_ptr, *inactive_vars_ptr);
          }
        },
        make_not_null(&box));
    if (cell_is_troubled) {
      // Set variables on subcells.
      if constexpr (System::has_primitive_and_conservative_vars) {
        db::mutate<typename System::primitive_variables_tag>(
            make_not_null(&box),
            [&subcell_mesh](const auto prim_vars_ptr) noexcept {
              prim_vars_ptr->initialize(subcell_mesh.number_of_grid_points());
            });
      }
      evolution::Initialization::Actions::
          SetVariables<Tags::Coordinates<Dim, Frame::Logical>>::apply(
              box, inboxes, cache, array_index, ActionList{},
              std::add_pointer_t<ParallelComponent>{nullptr});
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace evolution::dg::subcell::Actions
