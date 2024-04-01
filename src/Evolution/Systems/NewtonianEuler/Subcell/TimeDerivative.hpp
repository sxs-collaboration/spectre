// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
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
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
    using metavariables = typename std::decay_t<decltype(
        db::get<Parallel::Tags::Metavariables>(*box))>;
    using system = typename metavariables::system;
    using evolved_vars_tag = typename system::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using prim_tags = typename system::primitive_variables_tag::tags_list;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<Dim>, Frame::Inertial>;

    ASSERT((db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
                Dim, Frame::Grid, Frame::Inertial>>(*box))
               .is_identity(),
           "Do not yet support moving mesh with DG-subcell.");

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
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    const size_t reconstructed_num_pts =
        (subcell_mesh.extents(0) + 1) *
        subcell_mesh.extents().slice_away(0).product();

    const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
        cell_centered_logical_coords =
            db::get<evolution::dg::subcell::Tags::Coordinates<
                Dim, Frame::ElementLogical>>(*box);
    std::array<double, Dim> one_over_delta_xi{};
    for (size_t i = 0; i < Dim; ++i) {
      // Note: assumes isotropic extents
      gsl::at(one_over_delta_xi, i) =
          1.0 / (get<0>(cell_centered_logical_coords)[1] -
                 get<0>(cell_centered_logical_coords)[0]);
    }

    const NewtonianEuler::fd::Reconstructor<Dim>& recons =
        db::get<NewtonianEuler::fd::Tags::Reconstructor<Dim>>(*box);

    const Element<Dim>& element = db::get<domain::Tags::Element<Dim>>(*box);
    ASSERT(element.external_boundaries().size() == 0,
           "Can't have external boundaries right now with subcell. ElementID "
               << element.id());

    // Now package the data and compute the correction
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<system>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    std::array<Variables<evolved_vars_tags>, Dim> boundary_corrections{};
    tmpl::for_each<derived_boundary_corrections>([&boundary_correction,
                                                  &reconstructed_num_pts,
                                                  &recons, &box, &element,
                                                  &subcell_mesh,
                                                  &boundary_corrections](
                                                     auto
                                                         derived_correction_v) {
      using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
      if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
        using dg_package_data_temporary_tags =
            typename DerivedCorrection::dg_package_data_temporary_tags;
        using dg_package_data_argument_tags =
            tmpl::append<evolved_vars_tags, prim_tags, fluxes_tags,
                         dg_package_data_temporary_tags>;
        // Computed prims and cons on face via reconstruction
        auto package_data_argvars_lower_face = make_array<Dim>(
            Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
        auto package_data_argvars_upper_face = make_array<Dim>(
            Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

        // Reconstruct data to the face
        call_with_dynamic_type<void, typename NewtonianEuler::fd::Reconstructor<
                                         Dim>::creatable_classes>(
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

        using dg_package_field_tags =
            typename DerivedCorrection::dg_package_field_tags;
        // Allocated outside for loop to reduce allocations
        Variables<dg_package_field_tags> upper_packaged_data{
            reconstructed_num_pts};
        Variables<dg_package_field_tags> lower_packaged_data{
            reconstructed_num_pts};

        // Compute fluxes on faces
        for (size_t i = 0; i < Dim; ++i) {
          auto& vars_upper_face = gsl::at(package_data_argvars_upper_face, i);
          auto& vars_lower_face = gsl::at(package_data_argvars_lower_face, i);
          NewtonianEuler::subcell::compute_fluxes<Dim>(
              make_not_null(&vars_upper_face));
          NewtonianEuler::subcell::compute_fluxes<Dim>(
              make_not_null(&vars_lower_face));

          tnsr::i<DataVector, Dim, Frame::Inertial> lower_outward_conormal{
              reconstructed_num_pts, 0.0};
          lower_outward_conormal.get(i) = 1.0;

          tnsr::i<DataVector, Dim, Frame::Inertial> upper_outward_conormal{
              reconstructed_num_pts, 0.0};
          upper_outward_conormal.get(i) = -1.0;

          // Compute the packaged data
          using dg_package_data_projected_tags = tmpl::append<
              evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
              typename DerivedCorrection::dg_package_data_primitive_tags>;
          evolution::dg::Actions::detail::dg_package_data<system>(
              make_not_null(&upper_packaged_data),
              dynamic_cast<const DerivedCorrection&>(boundary_correction),
              vars_upper_face, upper_outward_conormal, {std::nullopt}, *box,
              typename DerivedCorrection::dg_package_data_volume_tags{},
              dg_package_data_projected_tags{});

          evolution::dg::Actions::detail::dg_package_data<system>(
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
              db::get<evolution::dg::Tags::MortarData<Dim>>(*box), 0);

          // Compute the corrections on the faces. We only need to
          // compute this once because we can just flip the normal
          // vectors then
          gsl::at(boundary_corrections, i).initialize(reconstructed_num_pts);
          evolution::dg::subcell::compute_boundary_terms(
              make_not_null(&gsl::at(boundary_corrections, i)),
              dynamic_cast<const DerivedCorrection&>(boundary_correction),
              upper_packaged_data, lower_packaged_data, db::as_access(*box),
              typename DerivedCorrection::dg_boundary_terms_volume_tags{});
        }
      }
    });

    // Now compute the actual time derivatives.
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, evolved_vars_tag>;
    using source_argument_tags = tmpl::list<
        Tags::MassDensityCons, Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
        hydro::Tags::SpatialVelocity<DataVector, Dim>,
        hydro::Tags::Pressure<DataVector>,
        hydro::Tags::SpecificInternalEnergy<DataVector>,
        hydro::Tags::EquationOfState<false, 2>,
        evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>,
        ::Tags::Time, NewtonianEuler::Tags::SourceTerm<Dim>>;
    db::mutate_apply<tmpl::list<dt_variables_tag>, source_argument_tags>(
        [&num_pts, &boundary_corrections, &subcell_mesh, &one_over_delta_xi,
         &cell_centered_logical_to_grid_inv_jacobian =
             db::get<evolution::dg::subcell::fd::Tags::
                         InverseJacobianLogicalToGrid<Dim>>(*box)](
            const auto dt_vars_ptr, const Scalar<DataVector>& mass_density_cons,
            const tnsr::I<DataVector, Dim>& momentum_density,
            const Scalar<DataVector>& energy_density,
            const tnsr::I<DataVector, Dim>& velocity,
            const Scalar<DataVector>& pressure,
            const Scalar<DataVector>& specific_internal_energy,
            const EquationsOfState::EquationOfState<false, 2>& eos,
            const tnsr::I<DataVector, Dim>& coords, const double time,
            const Sources::Source<Dim>& source) {
          dt_vars_ptr->initialize(num_pts, 0.0);
          using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
          using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;
          using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;

          auto& dt_mass = get<::Tags::dt<MassDensityCons>>(*dt_vars_ptr);
          auto& dt_momentum = get<::Tags::dt<MomentumDensity>>(*dt_vars_ptr);
          auto& dt_energy = get<::Tags::dt<EnergyDensity>>(*dt_vars_ptr);

          const auto eos_2d = eos.promote_to_2d_eos();
          source(make_not_null(&dt_mass), make_not_null(&dt_momentum),
                 make_not_null(&dt_energy), mass_density_cons, momentum_density,
                 energy_density, velocity, pressure, specific_internal_energy,
                 *eos_2d, coords, time);

          for (size_t dim = 0; dim < Dim; ++dim) {
            Scalar<DataVector>& mass_density_correction =
                get<MassDensityCons>(gsl::at(boundary_corrections, dim));
            Scalar<DataVector>& energy_density_correction =
                get<EnergyDensity>(gsl::at(boundary_corrections, dim));
            tnsr::I<DataVector, Dim, Frame::Inertial>&
                momentum_density_correction =
                    get<MomentumDensity>(gsl::at(boundary_corrections, dim));
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_mass)), gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(mass_density_correction), subcell_mesh.extents(), dim);
            evolution::dg::subcell::add_cartesian_flux_divergence(
                make_not_null(&get(dt_energy)), gsl::at(one_over_delta_xi, dim),
                cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                get(energy_density_correction), subcell_mesh.extents(), dim);
            for (size_t d = 0; d < Dim; ++d) {
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_momentum.get(d)),
                  gsl::at(one_over_delta_xi, dim),
                  cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                  momentum_density_correction.get(d), subcell_mesh.extents(),
                  dim);
            }
          }
        },
        box);
  }

 private:
  template <typename SourceTerm, typename... SourceTermArgs,
            typename... SourcedVars>
  static void sources_impl(std::tuple<gsl::not_null<Scalar<DataVector>*>,
                                      gsl::not_null<tnsr::I<DataVector, Dim>*>,
                                      gsl::not_null<Scalar<DataVector>*>>
                               dt_vars,
                           tmpl::list<SourcedVars...> /*meta*/,
                           const SourceTerm& source,
                           const SourceTermArgs&... source_term_args) {
    using dt_vars_list =
        tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                   Tags::EnergyDensity>;

    source.apply(
        std::get<tmpl::index_of<dt_vars_list, SourcedVars>::value>(dt_vars)...,
        source_term_args...);
  }
};
}  // namespace NewtonianEuler::subcell
