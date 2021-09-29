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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
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
      const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>&
          cell_centered_logical_to_grid_inv_jacobian,
      const Scalar<DataVector>& /*cell_centered_det_inv_jacobian*/) noexcept {
    using evolved_vars_tag = typename System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using prim_tags = typename System::primitive_variables_tag::tags_list;
    using recons_prim_tags = tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<3>, Frame::Inertial>;

    // The copy of Mesh is intentional to avoid a GCC-7 internal compiler error.
    const Mesh<3> subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
    ASSERT(
        subcell_mesh == Mesh<3>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                                subcell_mesh.quadrature(0)),
        "The subcell/FD mesh must be isotropic for the FD time derivative but "
        "got "
            << subcell_mesh);
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    const size_t reconstructed_num_pts =
        (subcell_mesh.extents(0) + 1) *
        subcell_mesh.extents().slice_away(0).product();

    const tnsr::I<DataVector, 3, Frame::ElementLogical>&
        cell_centered_logical_coords =
            db::get<evolution::dg::subcell::Tags::Coordinates<
                3, Frame::ElementLogical>>(*box);
    std::array<double, 3> one_over_delta_xi{};
    for (size_t i = 0; i < 3; ++i) {
      // Note: assumes isotropic extents
      gsl::at(one_over_delta_xi, i) =
          1.0 / (get<0>(cell_centered_logical_coords)[1] -
                 get<0>(cell_centered_logical_coords)[0]);
    }

    const grmhd::ValenciaDivClean::fd::Reconstructor& recons =
        db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(*box);

    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);
    ASSERT(element.external_boundaries().size() == 0,
           "Can't have external boundaries right now with subcell. ElementID "
               << element.id());

    // Now package the data and compute the correction
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    std::array<Variables<evolved_vars_tags>, 3> boundary_corrections{};
    tmpl::for_each<
        derived_boundary_corrections>([&](auto derived_correction_v) noexcept {
      using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
      if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
        using dg_package_data_temporary_tags =
            typename DerivedCorrection::dg_package_data_temporary_tags;
        using dg_package_data_argument_tags = tmpl::append<
            evolved_vars_tags, recons_prim_tags, fluxes_tags,
            tmpl::remove_duplicates<tmpl::push_back<
                dg_package_data_temporary_tags, gr::Tags::SpatialMetric<3>,
                gr::Tags::SqrtDetSpatialMetric<DataVector>,
                gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
                evolution::dg::Actions::detail::NormalVector<3>>>>;
        // Computed prims and cons on face via reconstruction
        auto vars_on_lower_face = make_array<3>(
            Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
        auto vars_on_upper_face = make_array<3>(
            Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
        // Copy over the face values of the metric quantities.
        using spacetime_vars_to_copy = tmpl::list<
            gr::Tags::Lapse<DataVector>,
            gr::Tags::Shift<3, Frame::Inertial, DataVector>,
            gr::Tags::SpatialMetric<3>,
            gr::Tags::SqrtDetSpatialMetric<DataVector>,
            gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>;
        tmpl::for_each<spacetime_vars_to_copy>(
            [&vars_on_lower_face, &vars_on_upper_face,
             &spacetime_vars_on_face =
                 db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                     typename System::flux_spacetime_variables_tag, 3>>(*box)](
                auto tag_v) noexcept {
              using tag = tmpl::type_from<decltype(tag_v)>;
              for (size_t d = 0; d < 3; ++d) {
                get<tag>(gsl::at(vars_on_lower_face, d)) =
                    get<tag>(gsl::at(spacetime_vars_on_face, d));
                get<tag>(gsl::at(vars_on_upper_face, d)) =
                    get<tag>(gsl::at(spacetime_vars_on_face, d));
              }
            });

        // Reconstruct data to the face
        call_with_dynamic_type<void, typename grmhd::ValenciaDivClean::fd::
                                         Reconstructor::creatable_classes>(
            &recons, [&box, &vars_on_lower_face,
                      &vars_on_upper_face](const auto& reconstructor) noexcept {
              db::apply<typename std::decay_t<decltype(
                  *reconstructor)>::reconstruction_argument_tags>(
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
        for (size_t i = 0; i < 3; ++i) {
          auto& vars_upper_face = gsl::at(vars_on_upper_face, i);
          auto& vars_lower_face = gsl::at(vars_on_lower_face, i);
          grmhd::ValenciaDivClean::subcell::compute_fluxes(
              make_not_null(&vars_upper_face));
          grmhd::ValenciaDivClean::subcell::compute_fluxes(
              make_not_null(&vars_lower_face));

          // Normal vectors in curved spacetime normalized by inverse
          // spatial metric. Since we assume a Cartesian grid, this is
          // relatively easy. Note that we use the sign convention on
          // the normal vectors to be compatible with DG.
          //
          // Note that these normal vectors are on all faces inside the DG
          // element since there are a bunch of subcells. We don't use the
          // NormalCovectorAndMagnitude tag in the DataBox right now to avoid
          // conflicts with the DG solver. We can explore in the future if it's
          // possible to reuse that allocation.
          const Scalar<DataVector> normalization{sqrt(
              get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
                                                 DataVector>>(vars_upper_face)
                  .get(i, i))};

          tnsr::i<DataVector, 3, Frame::Inertial> lower_outward_conormal{
              reconstructed_num_pts, 0.0};
          lower_outward_conormal.get(i) = 1.0 / get(normalization);

          tnsr::i<DataVector, 3, Frame::Inertial> upper_outward_conormal{
              reconstructed_num_pts, 0.0};
          upper_outward_conormal.get(i) = -lower_outward_conormal.get(i);
          // Note: we probably should compute the normal vector in addition to
          // the co-vector. Not a huge issue since we'll get an FPE right now if
          // it's used by a Riemann solver.

          // Compute the packaged data
          using dg_package_data_projected_tags = tmpl::append<
              evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
              typename DerivedCorrection::dg_package_data_primitive_tags>;
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
              make_not_null(&upper_packaged_data), i, element, subcell_mesh,
              db::get<evolution::dg::Tags::MortarData<3>>(*box));

          // Compute the corrections on the faces. We only need to
          // compute this once because we can just flip the normal
          // vectors then
          gsl::at(boundary_corrections, i).initialize(reconstructed_num_pts);
          evolution::dg::subcell::compute_boundary_terms(
              make_not_null(&gsl::at(boundary_corrections, i)),
              dynamic_cast<const DerivedCorrection&>(boundary_correction),
              upper_packaged_data, lower_packaged_data);
          // We need to multiply by the normal vector normalization
          gsl::at(boundary_corrections, i) *= get(normalization);
        }
      }
    });

    // Now compute the actual time derivatives.
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
    db::mutate_apply<
        tmpl::list<dt_variables_tag>,
        tmpl::list<
            grmhd::ValenciaDivClean::Tags::TildeD,
            grmhd::ValenciaDivClean::Tags::TildeTau,
            grmhd::ValenciaDivClean::Tags::TildeS<>,
            grmhd::ValenciaDivClean::Tags::TildeB<>,
            grmhd::ValenciaDivClean::Tags::TildePhi,
            hydro::Tags::SpatialVelocity<DataVector, 3>,
            hydro::Tags::MagneticField<DataVector, 3>,
            hydro::Tags::RestMassDensity<DataVector>,
            hydro::Tags::SpecificEnthalpy<DataVector>,
            hydro::Tags::LorentzFactor<DataVector>,
            hydro::Tags::Pressure<DataVector>, gr::Tags::Lapse<>,
            ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                          Frame::Inertial>,
            ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            gr::Tags::SpatialMetric<3>,
            ::Tags::deriv<
                gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                tmpl::size_t<3>, Frame::Inertial>,
            gr::Tags::InverseSpatialMetric<3>, gr::Tags::SqrtDetSpatialMetric<>,
            gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
            grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>>(
        [&cell_centered_logical_to_grid_inv_jacobian, &num_pts,
         &boundary_corrections, &subcell_mesh, &one_over_delta_xi](
            const auto dt_vars_ptr, const auto&... source_args) noexcept {
          dt_vars_ptr->initialize(num_pts, 0.0);
          using TildeD = grmhd::ValenciaDivClean::Tags::TildeD;
          using TildeTau = grmhd::ValenciaDivClean::Tags::TildeTau;
          using TildeS = grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>;
          using TildeB = grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>;
          using TildePhi = grmhd::ValenciaDivClean::Tags::TildePhi;

          auto& dt_tilde_d = get<::Tags::dt<TildeD>>(*dt_vars_ptr);
          auto& dt_tilde_tau = get<::Tags::dt<TildeTau>>(*dt_vars_ptr);
          auto& dt_tilde_s = get<::Tags::dt<TildeS>>(*dt_vars_ptr);
          auto& dt_tilde_b = get<::Tags::dt<TildeB>>(*dt_vars_ptr);
          auto& dt_tilde_phi = get<::Tags::dt<TildePhi>>(*dt_vars_ptr);

          grmhd::ValenciaDivClean::ComputeSources::apply(
              make_not_null(&dt_tilde_tau), make_not_null(&dt_tilde_s),
              make_not_null(&dt_tilde_b), make_not_null(&dt_tilde_phi),
              source_args...);

          for (size_t dim = 0; dim < 3; ++dim) {
            Scalar<DataVector>& tilde_d_density_correction =
                get<TildeD>(gsl::at(boundary_corrections, dim));
            Scalar<DataVector>& tilde_tau_density_correction =
                get<TildeTau>(gsl::at(boundary_corrections, dim));
            tnsr::i<DataVector, 3, Frame::Inertial>&
                tilde_s_density_correction =
                    get<TildeS>(gsl::at(boundary_corrections, dim));
            tnsr::I<DataVector, 3, Frame::Inertial>&
                tilde_b_density_correction =
                    get<TildeB>(gsl::at(boundary_corrections, dim));
            Scalar<DataVector>& tilde_phi_density_correction =
                get<TildePhi>(gsl::at(boundary_corrections, dim));
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_tilde_d)),
                gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(tilde_d_density_correction), subcell_mesh.extents(), dim);
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_tilde_tau)),
                gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(tilde_tau_density_correction), subcell_mesh.extents(), dim);
            for (size_t d = 0; d < 3; ++d) {
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_tilde_s.get(d)),
                  gsl::at(one_over_delta_xi, dim),
                  cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                  tilde_s_density_correction.get(d), subcell_mesh.extents(),
                  dim);
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_tilde_b.get(d)),
                  gsl::at(one_over_delta_xi, dim),
                  cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                  tilde_b_density_correction.get(d), subcell_mesh.extents(),
                  dim);
            }
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_tilde_phi)),
                gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(tilde_phi_density_correction), subcell_mesh.extents(), dim);
          }
        },
        box);
  }
};
}  // namespace grmhd::ValenciaDivClean::subcell
