// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace ScalarAdvection::subcell {
/*!
 * \brief Compute the time derivative on the subcell grid using FD
 * reconstruction.
 *
 * The code makes the following unchecked assumptions:
 * - Assumes Cartesian coordinates with a diagonal Jacobian matrix
 * from the logical to the inertial frame
 * - Assumes the mesh is not moving (grid and inertial frame are the same)
 */
template <size_t Dim>
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Grid>&
          cell_centered_logical_to_grid_inv_jacobian,
      const Scalar<DataVector>& /*cell_centered_det_inv_jacobian*/) {
    // subcell is currently not supported for external boundary elements
    const Element<Dim>& element = db::get<domain::Tags::Element<Dim>>(*box);
    ASSERT(element.external_boundaries().size() == 0,
           "Can't have external boundaries right now with subcell. ElementID "
               << element.id());

    using evolved_vars_tags = typename System<Dim>::variables_tag::tags_list;
    using fluxes_tags = typename Fluxes<Dim>::return_tags;

    // The copy of Mesh is intentional to avoid a GCC-7 internal compiler error.
    const Mesh<Dim> subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box);
    ASSERT(
        subcell_mesh == Mesh<Dim>(subcell_mesh.extents(0),
                                  subcell_mesh.basis(0),
                                  subcell_mesh.quadrature(0)),
        "The subcell/FD mesh must be isotropic for the FD time derivative but "
        "got "
            << subcell_mesh);

    const size_t num_reconstructed_pts =
        (subcell_mesh.extents(0) + 1) *
        subcell_mesh.extents().slice_away(0).product();

    const ScalarAdvection::fd::Reconstructor<Dim>& recons =
        db::get<ScalarAdvection::fd::Tags::Reconstructor<Dim>>(*box);

    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System<Dim>>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

    // Variables to store the boundary correction terms on FD subinterfaces
    std::array<Variables<evolved_vars_tags>, Dim> fd_boundary_corrections{};

    // package the data and compute the boundary correction
    tmpl::for_each<derived_boundary_corrections>([&boundary_correction,
                                                  &fd_boundary_corrections,
                                                  &box, &element,
                                                  &num_reconstructed_pts,
                                                  &recons, &subcell_mesh](
                                                     auto
                                                         derived_correction_v) {
      using derived_correction =
          tmpl::type_from<decltype(derived_correction_v)>;
      if (typeid(boundary_correction) == typeid(derived_correction)) {
        using dg_package_field_tags =
            typename derived_correction::dg_package_field_tags;
        using dg_package_data_temporary_tags =
            typename derived_correction::dg_package_data_temporary_tags;
        using dg_package_data_argument_tags =
            tmpl::append<evolved_vars_tags, fluxes_tags,
                         dg_package_data_temporary_tags>;

        // Variables that need to be reconstructed for dg_package_data()
        auto package_data_argvars_lower_face = make_array<Dim>(
            Variables<dg_package_data_argument_tags>(num_reconstructed_pts));
        auto package_data_argvars_upper_face = make_array<Dim>(
            Variables<dg_package_data_argument_tags>(num_reconstructed_pts));

        // Reconstruct the fields on interfaces
        call_with_dynamic_type<void, typename ScalarAdvection::fd::
                                         Reconstructor<Dim>::creatable_classes>(
            &recons,
            [&box, &package_data_argvars_lower_face,
             &package_data_argvars_upper_face](const auto& reconstructor) {
              db::apply<typename std::decay_t<decltype(
                  *reconstructor)>::reconstruction_argument_tags>(
                  [&package_data_argvars_lower_face,
                   &package_data_argvars_upper_face,
                   &reconstructor](const auto&... args) {
                    reconstructor->reconstruct(
                        make_not_null(&package_data_argvars_lower_face),
                        make_not_null(&package_data_argvars_upper_face),
                        args...);
                  },
                  *box);
            });

        // copy over the face values of the velocity field
        {
          using tag = Tags::VelocityField<Dim>;
          const auto& velocity_on_face =
              db::get<evolution::dg::subcell::Tags::OnSubcellFaces<tag, Dim>>(
                  *box);
          for (size_t d = 0; d < Dim; ++d) {
            get<tag>(gsl::at(package_data_argvars_lower_face, d)) =
                gsl::at(velocity_on_face, d);
            get<tag>(gsl::at(package_data_argvars_upper_face, d)) =
                gsl::at(velocity_on_face, d);
          }
        }

        // Variables to store packaged data. Allocate outside of loop to
        // reduce allocations
        Variables<dg_package_field_tags> packaged_data_upper_face{
            num_reconstructed_pts};
        Variables<dg_package_field_tags> packaged_data_lower_face{
            num_reconstructed_pts};

        for (size_t dim = 0; dim < Dim; ++dim) {
          auto& vars_upper_face = gsl::at(package_data_argvars_upper_face, dim);
          auto& vars_lower_face = gsl::at(package_data_argvars_lower_face, dim);

          // Compute fluxes on each faces
          ScalarAdvection::subcell::compute_fluxes<Dim>(
              make_not_null(&vars_upper_face));
          ScalarAdvection::subcell::compute_fluxes<Dim>(
              make_not_null(&vars_lower_face));

          // Note that we use the sign convention on the normal vectors to
          // be compatible with DG.
          tnsr::i<DataVector, Dim, Frame::Inertial> lower_outward_conormal{
              num_reconstructed_pts, 0.0};
          lower_outward_conormal.get(dim) = 1.0;
          tnsr::i<DataVector, Dim, Frame::Inertial> upper_outward_conormal{
              num_reconstructed_pts, 0.0};
          upper_outward_conormal.get(dim) = -1.0;

          // Compute the packaged data
          evolution::dg::Actions::detail::dg_package_data<System<Dim>>(
              make_not_null(&packaged_data_upper_face),
              dynamic_cast<const derived_correction&>(boundary_correction),
              vars_upper_face, upper_outward_conormal, {std::nullopt}, *box,
              typename derived_correction::dg_package_data_volume_tags{},
              dg_package_data_argument_tags{});
          evolution::dg::Actions::detail::dg_package_data<System<Dim>>(
              make_not_null(&packaged_data_lower_face),
              dynamic_cast<const derived_correction&>(boundary_correction),
              vars_lower_face, lower_outward_conormal, {std::nullopt}, *box,
              typename derived_correction::dg_package_data_volume_tags{},
              dg_package_data_argument_tags{});

          // Now need to check if any of our neighbors are doing DG, because
          // if so then we need to use whatever boundary data they sent
          // instead of what we computed locally.
          //
          // Note: We could check this beforehand to avoid the extra work of
          // reconstruction and flux computations at the boundaries.
          evolution::dg::subcell::correct_package_data<true>(
              make_not_null(&packaged_data_lower_face),
              make_not_null(&packaged_data_upper_face), dim, element,
              subcell_mesh,
              db::get<evolution::dg::Tags::MortarData<Dim>>(*box));

          // Compute the corrections on the faces. We only need to compute this
          // once because we can just flip the normal vectors then
          gsl::at(fd_boundary_corrections, dim)
              .initialize(num_reconstructed_pts);
          evolution::dg::subcell::compute_boundary_terms(
              make_not_null(&gsl::at(fd_boundary_corrections, dim)),
              dynamic_cast<const derived_correction&>(boundary_correction),
              packaged_data_upper_face, packaged_data_lower_face);
        }
      }
    });

    std::array<double, Dim> one_over_delta_xi{};
    {
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
          cell_centered_logical_coords =
              db::get<evolution::dg::subcell::Tags::Coordinates<
                  Dim, Frame::ElementLogical>>(*box);

      for (size_t i = 0; i < Dim; ++i) {
        // Note: assumes isotropic extents
        gsl::at(one_over_delta_xi, i) =
            1.0 / (get<0>(cell_centered_logical_coords)[1] -
                   get<0>(cell_centered_logical_coords)[0]);
      }
    }

    // Now compute the actual time derivative
    using dt_variables_tag =
        db::add_tag_prefix<::Tags::dt, typename System<Dim>::variables_tag>;
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    db::mutate<dt_variables_tag>(
        box, [&cell_centered_logical_to_grid_inv_jacobian, &num_pts,
              &fd_boundary_corrections, &subcell_mesh,
              &one_over_delta_xi](const auto dt_vars_ptr) {
          dt_vars_ptr->initialize(num_pts, 0.0);
          auto& dt_u = get<::Tags::dt<Tags::U>>(*dt_vars_ptr);

          for (size_t dim = 0; dim < Dim; ++dim) {
            Scalar<DataVector>& u_correction = get<::ScalarAdvection ::Tags::U>(
                gsl::at(fd_boundary_corrections, dim));
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_u)), gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(u_correction), subcell_mesh.extents(), dim);
          }
        });
  }
};
}  // namespace ScalarAdvection::subcell
