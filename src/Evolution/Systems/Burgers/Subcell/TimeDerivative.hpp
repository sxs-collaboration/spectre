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
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
      const InverseJacobian<DataVector, 1, Frame::Logical, Frame::Grid>&
          cell_centered_logical_to_grid_inv_jacobian,
      const Scalar<DataVector>& /*cell_centered_det_inv_jacobian*/) noexcept {
    using evolved_vars_tag = typename System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<1>, Frame::Inertial>;

    // The copy of Mesh is intentional to avoid a GCC-7 internal compiler error.
    const Mesh<1> subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<1>>(*box);
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    const size_t reconstructed_num_pts = num_pts + 1;

    const tnsr::I<DataVector, 1, Frame::Logical>& cell_centered_logical_coords =
        db::get<evolution::dg::subcell::Tags::Coordinates<1, Frame::Logical>>(
            *box);
    const double one_over_delta_xi =
        1.0 / (get<0>(cell_centered_logical_coords)[1] -
               get<0>(cell_centered_logical_coords)[0]);

    const Burgers::fd::Reconstructor& recons =
        db::get<Burgers::fd::Tags::Reconstructor>(*box);

    const Element<1>& element = db::get<domain::Tags::Element<1>>(*box);
    ASSERT(element.external_boundaries().size() == 0,
           "Can't have external boundaries right now with subcell. ElementID "
               << element.id());

    // Now package the data and compute the correction
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    std::array<Variables<evolved_vars_tags>, 1> boundary_corrections{};
    tmpl::for_each<
        derived_boundary_corrections>([&](auto derived_correction_v) noexcept {
      using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
      if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
        using dg_package_data_temporary_tags =
            typename DerivedCorrection::dg_package_data_temporary_tags;
        using dg_package_data_argument_tags =
            tmpl::append<evolved_vars_tags, fluxes_tags,
                         dg_package_data_temporary_tags>;
        // Computed prims and cons on face via reconstruction
        auto vars_on_lower_face = make_array<1>(
            Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
        auto vars_on_upper_face = make_array<1>(
            Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

        // Reconstruct data to the face
        call_with_dynamic_type<
            void, typename Burgers::fd::Reconstructor::creatable_classes>(
            &recons, [&box, &vars_on_lower_face,
                      &vars_on_upper_face](const auto& reconstructor) noexcept {
              db::apply<typename std::decay_t<
                  decltype(*reconstructor)>::reconstruction_argument_tags>(
                  [&vars_on_lower_face, &vars_on_upper_face,
                   &reconstructor](const auto&... args) noexcept {
                    reconstructor->reconstruct(
                        make_not_null(&vars_on_lower_face),
                        make_not_null(&vars_on_upper_face), args...);
                  },
                  *box);
            });

        using dg_package_field_tags =
            typename DerivedCorrection::dg_package_field_tags;
        // Allocated outside for loop to reduce allocations
        Variables<dg_package_field_tags> upper_packaged_data{
            reconstructed_num_pts};
        Variables<dg_package_field_tags> lower_packaged_data{
            reconstructed_num_pts};

        // Compute fluxes on faces
        auto& vars_upper_face = vars_on_upper_face[0];
        auto& vars_lower_face = vars_on_lower_face[0];
        Burgers::subcell::compute_fluxes(make_not_null(&vars_upper_face));
        Burgers::subcell::compute_fluxes(make_not_null(&vars_lower_face));

        // Note that we use the sign convention on the normal vectors to be
        // compatible with DG.
        const tnsr::i<DataVector, 1, Frame::Inertial> lower_outward_conormal{
            reconstructed_num_pts, 1.0};

        const tnsr::i<DataVector, 1, Frame::Inertial> upper_outward_conormal{
            reconstructed_num_pts, -1.0};

        // Compute the packaged data
        using dg_package_data_projected_tags =
            tmpl::append<evolved_vars_tags, fluxes_tags,
                         dg_package_data_temporary_tags>;
        evolution::dg::Actions::detail::dg_package_data<System>(
            make_not_null(&upper_packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_upper_face, upper_outward_conormal, {std::nullopt}, *box,
            typename DerivedCorrection::dg_package_data_volume_tags{},
            dg_package_data_projected_tags{});

        evolution::dg::Actions::detail::dg_package_data<System>(
            make_not_null(&lower_packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_lower_face, lower_outward_conormal, {std::nullopt}, *box,
            typename DerivedCorrection::dg_package_data_volume_tags{},
            dg_package_data_projected_tags{});

        // Now need to check if any of our neighbors are doing DG,
        // because if so then we need to use whatever boundary data
        // they sent instead of what we computed locally.
        //
        // Note: We could check this beforehand to avoid the extra
        // work of reconstruction and flux computations at the
        // boundaries.
        evolution::dg::subcell::correct_package_data<true>(
            make_not_null(&lower_packaged_data),
            make_not_null(&upper_packaged_data), 0, element, subcell_mesh,
            db::get<evolution::dg::Tags::MortarData<1>>(*box));

        // Compute the corrections on the faces. We only need to
        // compute this once because we can just flip the normal
        // vectors then
        boundary_corrections[0].initialize(reconstructed_num_pts);
        evolution::dg::subcell::compute_boundary_terms(
            make_not_null(&(boundary_corrections[0])),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            upper_packaged_data, lower_packaged_data);
      }
    });

    // Now compute the actual time derivatives.
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
    db::mutate<dt_variables_tag>(
        box, [&cell_centered_logical_to_grid_inv_jacobian, &num_pts,
              &boundary_corrections, &subcell_mesh,
              &one_over_delta_xi](const auto dt_vars_ptr) noexcept {
          dt_vars_ptr->initialize(num_pts, 0.0);
          auto& dt_u = get<::Tags::dt<Burgers::Tags::U>>(*dt_vars_ptr);

          Scalar<DataVector>& u_correction =
              get<Burgers::Tags::U>(boundary_corrections[0]);
          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&get(dt_u)), one_over_delta_xi,
              get<0, 0>(cell_centered_logical_to_grid_inv_jacobian),
              get(u_correction), subcell_mesh.extents(), 0);
        });
  }
};
}  // namespace Burgers::subcell
