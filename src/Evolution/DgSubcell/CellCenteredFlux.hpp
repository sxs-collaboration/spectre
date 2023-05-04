// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
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
  using argument_tags = tmpl::push_front<
      typename FluxMutator::argument_tags, subcell::Tags::SubcellOptions<Dim>,
      subcell::Tags::Mesh<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::MeshVelocity<Dim, Frame::Inertial>,
      ::Tags::Variables<flux_variables>, subcell::Tags::DidRollback>;

  template <typename... FluxTags, typename... Args>
  static void apply(
      const gsl::not_null<std::optional<Variables<tmpl::list<FluxTags...>>>*>
          cell_centered_fluxes,
      const subcell::SubcellOptions& subcell_options,
      const Mesh<Dim>& subcell_mesh, const Mesh<Dim>& dg_mesh,
      const std::optional<tnsr::I<DataVector, Dim>>& dg_mesh_velocity,
      const ::Variables<flux_variables>& cell_centered_flux_vars,
      const bool did_rollback, Args&&... args) {
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
        if (dg_mesh_velocity.has_value()) {
          for (size_t i = 0; i < Dim; i++) {
            //
            // Project mesh velocity on face mesh. We only need the component
            // orthogonal to the face.
            const DataVector& cell_centered_mesh_velocity =
                evolution::dg::subcell::fd::project(
                    dg_mesh_velocity.value().get(i), dg_mesh,
                    subcell_mesh.extents());

            tmpl::for_each<flux_variables>(
                [&cell_centered_flux_vars, &cell_centered_mesh_velocity,
                 &cell_centered_fluxes, &i](auto tag_v) {
                  using tag = tmpl::type_from<decltype(tag_v)>;
                  using flux_tag =
                      ::Tags::Flux<tag, tmpl::size_t<Dim>, Frame::Inertial>;
                  using FluxTensor = typename flux_tag::type;
                  const auto& var = get<tag>(cell_centered_flux_vars);
                  auto& flux = get<flux_tag>(cell_centered_fluxes->value());
                  for (size_t storage_index = 0; storage_index < var.size();
                       ++storage_index) {
                    const auto tensor_index =
                        var.get_tensor_index(storage_index);
                    const auto flux_storage_index =
                        FluxTensor::get_storage_index(prepend(tensor_index, i));
                    flux[flux_storage_index] -=
                        cell_centered_mesh_velocity * var[storage_index];
                  }
                });
          }
        }
      }
    }
  }
};
}  // namespace evolution::dg::subcell::fd
