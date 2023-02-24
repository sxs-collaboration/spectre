// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::fd {
/*!
 * \brief Mutator that wraps the system's `FluxMutator` to correctly set the
 * cell-centered fluxes on the subcell grid.
 *
 * Currently we only use high-order FD if the FD order was specified in the
 * input file. We will need to extend this to support adaptive-order in the
 * future. In that case we need to check if the FD reconstruction reports back
 * the order to use.
 */
template <typename System, typename FluxMutator, size_t Dim,
          bool ComputeOnlyOnRollback, typename Fr = Frame::Inertial>
struct CellCenteredFlux {
  using flux_variables = typename System::flux_variables;

  using return_tags =
      tmpl::list<subcell::Tags::CellCenteredFlux<flux_variables, Dim>>;
  using argument_tags =
      tmpl::push_front<typename FluxMutator::argument_tags,
                       subcell::Tags::SubcellOptions<Dim>,
                       subcell::Tags::Mesh<Dim>, subcell::Tags::DidRollback>;

  template <typename... FluxTags, typename... Args>
  static void apply(
      const gsl::not_null<std::optional<Variables<tmpl::list<FluxTags...>>>*>
          cell_centered_fluxes,
      const subcell::SubcellOptions& subcell_options,
      const Mesh<Dim>& subcell_mesh, const bool did_rollback, Args&&... args) {
    if (did_rollback or not ComputeOnlyOnRollback) {
      if (subcell_options.finite_difference_derivative_order() !=
          ::fd::DerivativeOrder::Two) {
        if (not cell_centered_fluxes->has_value()) {
          (*cell_centered_fluxes) =
              Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                                         tmpl::size_t<Dim>, Fr>>{
                  subcell_mesh.number_of_grid_points()};
        }
        FluxMutator::apply(
            make_not_null(&get<FluxTags>((*cell_centered_fluxes).value()))...,
            std::forward<Args>(args)...);
      }
    }
  }
};
}  // namespace evolution::dg::subcell::fd
