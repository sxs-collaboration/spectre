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
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Burgers::subcell {
/*!
 * \brief Compute the time derivative on the subcell grid using FD
 * reconstruction.
 *
 * The code makes the following unchecked assumptions:
 * - Assumes Cartesian coordinates with a diagonal Jacobian matrix
 * from the logical to the inertial frame
 * - Assumes the mesh is not moving (grid and inertial frame are the same)
 */
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>&
          cell_centered_logical_to_grid_inv_jacobian,
      const Scalar<DataVector>& /*cell_centered_det_inv_jacobian*/) {
    const Element<1>& element = db::get<domain::Tags::Element<1>>(*box);

    const bool element_is_interior = element.external_boundaries().size() == 0;
    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::SubcellOptions::subcell_enabled_at_external_boundary;

    ASSERT(element_is_interior or subcell_enabled_at_external_boundary,
           "Subcell time derivative is called at a boundary element while "
           "using subcell is disabled at external boundaries."
           "ElementID "
               << element.id());

    using evolved_vars_tags = typename System::variables_tag::tags_list;
    using fluxes_tags = typename Fluxes::return_tags;

    // The copy of Mesh is intentional to avoid a GCC-7 internal compiler error.
    const Mesh<1> subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<1>>(*box);
    const size_t num_reconstructed_pts =
        subcell_mesh.number_of_grid_points() + 1;

    const Burgers::fd::Reconstructor& recons =
        db::get<Burgers::fd::Tags::Reconstructor>(*box);

    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    // Variables to store the boundary correction terms on FD subinterfaces
    std::array<Variables<evolved_vars_tags>, 1> fd_boundary_corrections{};

    // If the element has external boundaries and subcell is enabled for
    // boundary elements, compute FD ghost data with a given boundary condition.
    if constexpr (subcell_enabled_at_external_boundary) {
      if (element.external_boundaries().size() != 0) {
        fd::BoundaryConditionGhostData::apply(box, element, recons);
      }
    }

    // package the data and compute the boundary correction
    tmpl::for_each<derived_boundary_corrections>(
        [&boundary_correction, &box, &element, &fd_boundary_corrections,
         &num_reconstructed_pts, &recons,
         &subcell_mesh](auto derived_correction_v) {
          using derived_correction =
              tmpl::type_from<decltype(derived_correction_v)>;
          if (typeid(boundary_correction) == typeid(derived_correction)) {
            using dg_package_field_tags =
                typename derived_correction::dg_package_field_tags;
            using dg_package_data_argument_tags =
                tmpl::append<evolved_vars_tags, fluxes_tags>;

            // Variables that need to be reconstructed for dg_package_data()
            auto package_data_argvars_lower_face =
                make_array<1>(Variables<dg_package_data_argument_tags>(
                    num_reconstructed_pts));
            auto package_data_argvars_upper_face =
                make_array<1>(Variables<dg_package_data_argument_tags>(
                    num_reconstructed_pts));

            // Reconstruct the fields on interfaces
            call_with_dynamic_type<
                void, typename Burgers::fd::Reconstructor::creatable_classes>(
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

            // Variables to store packaged data. Allocate outside of loop to
            // reduce allocations
            Variables<dg_package_field_tags> upper_packaged_data{
                num_reconstructed_pts};
            Variables<dg_package_field_tags> lower_packaged_data{
                num_reconstructed_pts};

            auto& vars_upper_face = gsl::at(package_data_argvars_upper_face, 0);
            auto& vars_lower_face = gsl::at(package_data_argvars_lower_face, 0);

            // Compute fluxes on each faces
            Burgers::subcell::compute_fluxes(make_not_null(&vars_upper_face));
            Burgers::subcell::compute_fluxes(make_not_null(&vars_lower_face));

            // Note that we use the sign convention on the normal vectors to
            // be compatible with DG.
            tnsr::i<DataVector, 1, Frame::Inertial> lower_outward_conormal{
                num_reconstructed_pts, 0.0};
            lower_outward_conormal.get(0) = 1.0;
            tnsr::i<DataVector, 1, Frame::Inertial> upper_outward_conormal{
                num_reconstructed_pts, 0.0};
            upper_outward_conormal.get(0) = -1.0;

            // Compute the packaged data
            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&upper_packaged_data),
                dynamic_cast<const derived_correction&>(boundary_correction),
                vars_upper_face, upper_outward_conormal, {std::nullopt}, *box,
                typename derived_correction::dg_package_data_volume_tags{},
                dg_package_data_argument_tags{});
            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&lower_packaged_data),
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
                make_not_null(&lower_packaged_data),
                make_not_null(&upper_packaged_data), 0, element, subcell_mesh,
                db::get<evolution::dg::Tags::MortarData<1>>(*box), 0);

            // Compute the corrections on the faces. We only need to compute
            // this once because we can just flip the normal vectors then
            gsl::at(fd_boundary_corrections, 0)
                .initialize(num_reconstructed_pts);
            evolution::dg::subcell::compute_boundary_terms(
                make_not_null(&gsl::at(fd_boundary_corrections, 0)),
                dynamic_cast<const derived_correction&>(boundary_correction),
                upper_packaged_data, lower_packaged_data);
          }
        });

    std::array<double, 1> one_over_delta_xi{};
    {
      const tnsr::I<DataVector, 1, Frame::ElementLogical>&
          cell_centered_logical_coords =
              db::get<evolution::dg::subcell::Tags::Coordinates<
                  1, Frame::ElementLogical>>(*box);

      gsl::at(one_over_delta_xi, 0) =
          1.0 / (get<0>(cell_centered_logical_coords)[1] -
                 get<0>(cell_centered_logical_coords)[0]);
    }

    // Now compute the actual time derivative
    using dt_variables_tag =
        db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    db::mutate<dt_variables_tag>(
        [&cell_centered_logical_to_grid_inv_jacobian, &num_pts,
         &fd_boundary_corrections, &subcell_mesh,
         &one_over_delta_xi](const auto dt_vars_ptr) {
          dt_vars_ptr->initialize(num_pts, 0.0);
          auto& dt_u = get<::Tags::dt<::Burgers::Tags::U>>(*dt_vars_ptr);

          Scalar<DataVector>& u_correction =
              get<::Burgers::Tags::U>(gsl::at(fd_boundary_corrections, 0));
          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&get(dt_u)), gsl::at(one_over_delta_xi, 0),
              cell_centered_logical_to_grid_inv_jacobian.get(0, 0),
              get(u_correction), subcell_mesh.extents(), 0);
        },
        box);
  }
};
}  // namespace Burgers::subcell
