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
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FilterOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Filters.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
namespace detail {
template <class GhDtTagsList, class GhTemporariesList, class GhGradientTagsList,
          class GhExtraTagsList, class GrmhdDtTagsList,
          class GrmhdSourceTagsList, class GrmhdArgumentSourceTagsList>
struct ComputeTimeDerivImpl;

template <class... GhDtTags, class... GhTemporaries, class... GhGradientTags,
          class... GhExtraTags, class... GrmhdDtTags, class... GrmhdSourceTags,
          class... GrmhdArgumentSourceTags>
struct ComputeTimeDerivImpl<
    tmpl::list<GhDtTags...>, tmpl::list<GhTemporaries...>,
    tmpl::list<GhGradientTags...>, tmpl::list<GhExtraTags...>,
    tmpl::list<GrmhdDtTags...>, tmpl::list<GrmhdSourceTags...>,
    tmpl::list<GrmhdArgumentSourceTags...>> {
  template <class DbTagsList>
  static void apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>&
          cell_centered_logical_to_inertial_inv_jacobian,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>&
          cell_centered_logical_to_grid_inv_jacobian,
      const std::array<double, 3>& one_over_delta_xi,
      const std::array<Variables<tmpl::list<GrmhdDtTags...>>, 3>&
          boundary_corrections,
      const Variables<
          db::wrap_tags_in<::Tags::deriv, typename System::gradients_tags,
                           tmpl::size_t<3>, Frame::Inertial>>& gh_derivs) {
    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
    // Note: GH+GRMHD tags are always GH,GRMHD
    using deriv_lapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                      tmpl::size_t<3>, Frame::Inertial>;
    using deriv_shift = ::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>;
    using deriv_spatial_metric =
        ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>;
    using extra_tags_for_grmhd =
        tmpl::list<deriv_lapse, deriv_shift, deriv_spatial_metric,
                   gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
    using temporary_tags = tmpl::remove_duplicates<tmpl::append<
        typename gh::TimeDerivative<3_st>::temporary_tags,
        tmpl::push_front<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             temporary_tags,
                         ::gh::ConstraintDamping::Tags::ConstraintGamma0>,
        extra_tags_for_grmhd,
        tmpl::list<Tags::TraceReversedStressEnergy, Tags::FourVelocityOneForm,
                   Tags::ComovingMagneticFieldOneForm>>>;
    Variables<temporary_tags> temp_tags{subcell_mesh.number_of_grid_points()};
    const auto temp_tags_ptr = make_not_null(&temp_tags);

    // Compute constraint damping terms.
    const double time = db::get<::Tags::Time>(*box);
    const auto& functions_of_time =
        db::get<::domain::Tags::FunctionsOfTime>(*box);
    const auto& grid_coords =
        db::get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Grid>>(
            *box);
    db::get<
        gh::ConstraintDamping::Tags::DampingFunctionGamma0<3, Frame::Grid>> (
        *box)(get<gh::ConstraintDamping::Tags::ConstraintGamma0>(temp_tags_ptr),
              grid_coords, time, functions_of_time);
    db::get<
        gh::ConstraintDamping::Tags::DampingFunctionGamma1<3, Frame::Grid>> (
        *box)(get<gh::ConstraintDamping::Tags::ConstraintGamma1>(temp_tags_ptr),
              grid_coords, time, functions_of_time);
    db::get<
        gh::ConstraintDamping::Tags::DampingFunctionGamma2<3, Frame::Grid>> (
        *box)(get<gh::ConstraintDamping::Tags::ConstraintGamma2>(temp_tags_ptr),
              grid_coords, time, functions_of_time);

    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
    const gsl::not_null<typename dt_variables_tag::type*> dt_vars_ptr =
        db::mutate<dt_variables_tag>(
            [](const auto local_dt_vars_ptr) { return local_dt_vars_ptr; },
            box);
    dt_vars_ptr->initialize(subcell_mesh.number_of_grid_points());

    using primitives_tag = typename System::primitive_variables_tag;
    using evolved_vars_tag =
        typename grmhd::GhValenciaDivClean::System::variables_tag;

    const auto& primitive_vars = db::get<primitives_tag>(*box);
    const auto& evolved_vars = db::get<evolved_vars_tag>(*box);

    gh::TimeDerivative<3_st>::apply(
        get<::Tags::dt<GhDtTags>>(dt_vars_ptr)...,
        get<GhTemporaries>(temp_tags_ptr)...,
        get<::Tags::deriv<GhGradientTags, tmpl::size_t<3>, Frame::Inertial>>(
            gh_derivs)...,
        get<GhExtraTags>(evolved_vars, temp_tags)...,

        db::get<::gh::gauges::Tags::GaugeCondition>(*box),
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box), time,
        inertial_coords, cell_centered_logical_to_inertial_inv_jacobian,
        db::get<domain::Tags::MeshVelocity<3>>(*box));

    {
      // Set extra tags needed for GRMHD source terms. We compute these from
      // quantities already computed inside the GH RHS computation to minimize
      // FLOPs.
      const auto& lapse = get<gr::Tags::Lapse<DataVector>>(temp_tags);
      const auto& half_phi_two_normals =
          get<gh::Tags::HalfPhiTwoNormals<3>>(temp_tags);
      const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(evolved_vars);
      const auto& phi_one_normal = get<gh::Tags::PhiOneNormal<3>>(temp_tags);
      const auto& spacetime_normal_vector =
          get<gr::Tags::SpacetimeNormalVector<DataVector, 3>>(temp_tags);
      const auto& inverse_spacetime_metric =
          get<gr::Tags::InverseSpacetimeMetric<DataVector, 3>>(temp_tags);

      auto& spatial_deriv_lapse = get<deriv_lapse>(temp_tags);
      auto& spatial_deriv_shift = get<deriv_shift>(temp_tags);
      // Compute d_i beta^i
      for (size_t i = 0; i < 3; ++i) {
        // Use spatial_deriv_lapse as temp buffer to reduce number of 2*
        // operations.
        const auto& phi_two_normals_i = spatial_deriv_lapse.get(i) =
            2.0 * half_phi_two_normals.get(i);
        for (size_t j = 0; j < 3; ++j) {
          spatial_deriv_shift.get(i, j) =
              spacetime_normal_vector.get(j + 1) * phi_two_normals_i;
          for (size_t a = 0; a < 4; ++a) {
            spatial_deriv_shift.get(i, j) +=
                inverse_spacetime_metric.get(j + 1, a) *
                phi_one_normal.get(i, a);
          }
          spatial_deriv_shift.get(i, j) *= get(lapse);
        }
      }

      // Compute d_i lapse
      for (size_t i = 0; i < 3; ++i) {
        spatial_deriv_lapse.get(i) = -get(lapse) * half_phi_two_normals.get(i);
      }
      // Extract d_i \gamma_{ij}
      for (size_t k = 0; k < 3; ++k) {
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = i; j < 3; ++j) {
            get<deriv_spatial_metric>(temp_tags).get(k, i, j) =
                phi.get(k, i + 1, j + 1);
          }
        }
      }

      // Compute extrinsic curvature
      const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(evolved_vars);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(temp_tags).get(i,
                                                                          j) =
              0.5 * (pi.get(i + 1, j + 1) + phi_one_normal.get(i, j + 1) +
                     phi_one_normal.get(j, i + 1));
        }
      }
    }  // End scope for computing metric terms in GRMHD source terms.

    grmhd::ValenciaDivClean::ComputeSources::apply(
        get<::Tags::dt<GrmhdSourceTags>>(dt_vars_ptr)...,
        get<GrmhdArgumentSourceTags>(temp_tags, primitive_vars, evolved_vars,
                                     *box)...);

    tenex::evaluate<ti::i>(get<hydro::Tags::SpatialVelocityOneForm<
                               DataVector, 3, Frame::Inertial>>(temp_tags_ptr),
                           get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                               primitive_vars)(ti::J) *
                               get<gr::Tags::SpatialMetric<DataVector, 3>>(
                                   temp_tags)(ti::i, ti::j));

    tenex::evaluate<ti::i>(
        get<hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Inertial>>(
            temp_tags_ptr),
        get<hydro::Tags::MagneticField<DataVector, 3>>(primitive_vars)(ti::J) *
            get<gr::Tags::SpatialMetric<DataVector, 3>>(temp_tags)(ti::i,
                                                                   ti::j));

    tenex::evaluate(
        get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tags_ptr),
        get<hydro::Tags::MagneticField<DataVector, 3>>(primitive_vars)(ti::J) *
            get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tags)(
                ti::j));

    tenex::evaluate(
        get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
            temp_tags_ptr),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(primitive_vars)(
            ti::J) *
            get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tags)(
                ti::j));

    tenex::evaluate(get<typename ValenciaDivClean::TimeDerivativeTerms::
                            OneOverLorentzFactorSquared>(temp_tags_ptr),
                    1.0 / (square(get<hydro::Tags::LorentzFactor<DataVector>>(
                              primitive_vars)())));

    trace_reversed_stress_energy(
        get<Tags::TraceReversedStressEnergy>(temp_tags_ptr),
        get<Tags::FourVelocityOneForm>(temp_tags_ptr),
        get<Tags::ComovingMagneticFieldOneForm>(temp_tags_ptr),

        get<hydro::Tags::RestMassDensity<DataVector>>(evolved_vars, temp_tags,
                                                      primitive_vars),
        get<hydro::Tags::SpecificEnthalpy<DataVector>>(evolved_vars, temp_tags,
                                                       primitive_vars),
        get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3,
                                                Frame::Inertial>>(
            evolved_vars, temp_tags, primitive_vars),

        get<hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Inertial>>(
            evolved_vars, temp_tags, primitive_vars),

        get<hydro::Tags::MagneticFieldSquared<DataVector>>(
            evolved_vars, temp_tags, primitive_vars),

        get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
            evolved_vars, temp_tags, primitive_vars),
        get<hydro::Tags::LorentzFactor<DataVector>>(evolved_vars, temp_tags,
                                                    primitive_vars),
        get<typename ValenciaDivClean::TimeDerivativeTerms::
                OneOverLorentzFactorSquared>(evolved_vars, temp_tags,
                                             primitive_vars),
        get<hydro::Tags::Pressure<DataVector>>(evolved_vars, temp_tags,
                                               primitive_vars),
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved_vars, temp_tags,
                                                      primitive_vars),
        get<gr::Tags::Shift<DataVector, 3>>(evolved_vars, temp_tags,
                                            primitive_vars),
        get<gr::Tags::Lapse<DataVector>>(evolved_vars, temp_tags,
                                         primitive_vars));

    add_stress_energy_term_to_dt_pi(
        get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(dt_vars_ptr),
        get<Tags::TraceReversedStressEnergy>(temp_tags),
        get<gr::Tags::Lapse<DataVector>>(temp_tags));

    for (size_t dim = 0; dim < 3; ++dim) {
      const auto& boundary_correction_in_axis =
          gsl::at(boundary_corrections, dim);
      const auto& component_inverse_jacobian =
          cell_centered_logical_to_grid_inv_jacobian.get(dim, dim);
      const double inverse_delta = gsl::at(one_over_delta_xi, dim);
      EXPAND_PACK_LEFT_TO_RIGHT([&dt_vars_ptr, &boundary_correction_in_axis,
                                 &component_inverse_jacobian, dim,
                                 inverse_delta, &subcell_mesh]() {
        auto& dt_var = *get<::Tags::dt<GrmhdDtTags>>(dt_vars_ptr);
        const auto& var_correction =
            get<GrmhdDtTags>(boundary_correction_in_axis);
        for (size_t i = 0; i < dt_var.size(); ++i) {
          if constexpr (not tmpl::list_contains_v<
                            tmpl::list<GrmhdSourceTags...>, GrmhdDtTags>) {
            // On the first iteration of the loop over `dim`, zero the GRMHD
            // dt(u) for variables that do not have a source term . This is
            // necessary to avoid `+=` to a `NaN` (debug mode) or random garbage
            // (release mode). `add_cartesian_flux_divergence` does a `+=`
            // internally.
            if (dim == 0) {
              dt_var[i] = 0.0;
            }
          }

          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&dt_var[i]), inverse_delta,
              component_inverse_jacobian, var_correction[i],
              subcell_mesh.extents(), dim);
        }
      }());
    }
  }
};
}  // namespace detail

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
      const Scalar<DataVector>& /*cell_centered_det_inv_jacobian*/) {
    using evolved_vars_tag =
        typename grmhd::GhValenciaDivClean::System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using grmhd_evolved_vars_tag =
        typename grmhd::ValenciaDivClean::System::variables_tag;
    using grmhd_evolved_vars_tags = typename grmhd_evolved_vars_tag::tags_list;
    using fluxes_tags =
        db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                         tmpl::size_t<3>, Frame::Inertial>;
    using prim_tag = typename System::primitive_variables_tag;
    using prim_tags = typename prim_tag::tags_list;
    using recons_prim_tags = tmpl::push_front<tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>>;
    using gradients_tags = typename System::gradients_tags;

    const Mesh<3>& subcell_mesh =
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
    const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        cell_centered_logical_to_inertial_inv_jacobian{};
    ASSERT((db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
                3, Frame::Grid, Frame::Inertial>>(*box)
                .is_identity()),
           "Using time-dependent domain but this is not yet supported with "
           "DG-FD.");
    for (size_t i = 0;
         i < cell_centered_logical_to_inertial_inv_jacobian.size(); ++i) {
      make_const_view(
          make_not_null(&cell_centered_logical_to_inertial_inv_jacobian[i]),
          cell_centered_logical_to_grid_inv_jacobian[i], 0_st,
          cell_centered_logical_to_grid_inv_jacobian[i].size());
    }
    const auto& grid_coords =
        db::get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Grid>>(
            *box);
    const tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};
    for (size_t i = 0; i < grid_coords.size(); ++i) {
      make_const_view(make_not_null(&inertial_coords.get(i)),
                      grid_coords.get(i), 0_st, grid_coords.get(i).size());
    }

    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);
    const bool element_is_interior = element.external_boundaries().empty();
    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::SubcellOptions::subcell_enabled_at_external_boundary;

    ASSERT(element_is_interior or subcell_enabled_at_external_boundary,
           "Subcell time derivative is called at a boundary element while "
           "using subcell is disabled at external boundaries."
           "ElementID "
               << element.id());

    const fd::Reconstructor& recons = db::get<fd::Tags::Reconstructor>(*box);
    // If the element has external boundaries and subcell is enabled for
    // boundary elements, compute FD ghost data with a given boundary condition.
    if constexpr (subcell_enabled_at_external_boundary) {
      if (not element_is_interior) {
        fd::BoundaryConditionGhostData::apply(box, element, recons);
      }
    }

    if (const auto& filter_options =
            db::get<grmhd::GhValenciaDivClean::fd::Tags::FilterOptions>(*box);
        filter_options.spacetime_dissipation.has_value()) {
      db::mutate<evolved_vars_tag>(
          [&filter_options, &recons, &subcell_mesh](const auto evolved_vars_ptr,
                                                    const auto& ghost_data) {
            typename evolved_vars_tag::type filtered_vars = *evolved_vars_ptr;
            // $(recons.ghost_zone_size() - 1) * 2 + 1$ => always use highest
            // order dissipation filter possible.
            grmhd::GhValenciaDivClean::fd::spacetime_kreiss_oliger_filter(
                make_not_null(&filtered_vars), *evolved_vars_ptr, ghost_data,
                subcell_mesh, 2 * recons.ghost_zone_size(),
                filter_options.spacetime_dissipation.value());
            *evolved_vars_ptr = filtered_vars;
          },
          box,
          db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
              *box));
    }

    // GH+GRMHD is a bit different.
    // 1. Compute GH time derivative, since this will also give us lapse, shift,
    //    etc. that we need to reconstruct.
    // 2. Compute d_t Pi_{ab} source terms from MHD (or do we wait until post
    //    MHD source terms?)
    // 3. Reconstruct MHD+spacetime vars to interfaces
    // 4. Compute MHD time derivatives.
    //
    // Compute FD GH derivatives with neighbor data
    const auto& evolved_vars = db::get<evolved_vars_tag>(*box);
    Variables<db::wrap_tags_in<::Tags::deriv, gradients_tags, tmpl::size_t<3>,
                               Frame::Inertial>>
        cell_centered_gh_derivs{num_pts};
    grmhd::GhValenciaDivClean::fd::spacetime_derivatives(
        make_not_null(&cell_centered_gh_derivs), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            *box),
        subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

    // Now package the data and compute the correction
    //
    // Note: Assumes a the GH and GRMHD corrections can be invoked separately.
    // This is reasonable since the systems are a tensor product system.
    const auto& base_boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections = typename std::decay_t<decltype(
        base_boundary_correction)>::creatable_classes;
    std::array<Variables<grmhd_evolved_vars_tags>, 3> boundary_corrections{};
    call_with_dynamic_type<void, derived_boundary_corrections>(
        &base_boundary_correction, [&](const auto* gh_grmhd_correction) {
          // Need the GH packaged tags to avoid projecting them.
          using gh_dg_package_field_tags = typename std::decay_t<decltype(
              gh_grmhd_correction->gh_correction())>::dg_package_field_tags;
          // Only apply correction to GRMHD variables.
          const auto& boundary_correction =
              gh_grmhd_correction->valencia_correction();
          using DerivedCorrection = std::decay_t<decltype(boundary_correction)>;
          using dg_package_data_temporary_tags =
              typename DerivedCorrection::dg_package_data_temporary_tags;

          using dg_package_data_argument_tags = tmpl::append<
              evolved_vars_tags, recons_prim_tags, fluxes_tags,
              tmpl::remove_duplicates<tmpl::push_back<
                  dg_package_data_temporary_tags,
                  gr::Tags::SpatialMetric<DataVector, 3>,
                  gr::Tags::SqrtDetSpatialMetric<DataVector>,
                  gr::Tags::InverseSpatialMetric<DataVector, 3>,
                  evolution::dg::Actions::detail::NormalVector<3>>>>;

          // Computed prims and cons on face via reconstruction
          auto package_data_argvars_lower_face = make_array<3>(
              Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
          auto package_data_argvars_upper_face = make_array<3>(
              Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

          // Reconstruct data to the face
          call_with_dynamic_type<void, typename grmhd::GhValenciaDivClean::fd::
                                           Reconstructor::creatable_classes>(
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
          for (size_t i = 0; i < 3; ++i) {
            auto& vars_upper_face = gsl::at(package_data_argvars_upper_face, i);
            auto& vars_lower_face = gsl::at(package_data_argvars_lower_face, i);
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
            // conflicts with the DG solver. We can explore in the future if
            // it's possible to reuse that allocation.
            const Scalar<DataVector> normalization{
                sqrt(get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                         vars_upper_face)
                         .get(i, i))};

            tnsr::i<DataVector, 3, Frame::Inertial> lower_outward_conormal{
                reconstructed_num_pts, 0.0};
            lower_outward_conormal.get(i) = 1.0 / get(normalization);

            tnsr::i<DataVector, 3, Frame::Inertial> upper_outward_conormal{
                reconstructed_num_pts, 0.0};
            upper_outward_conormal.get(i) = -lower_outward_conormal.get(i);
            // Note: we probably should compute the normal vector in addition to
            // the co-vector. Not a huge issue since we'll get an FPE right now
            // if it's used by a Riemann solver.

            // Compute the packaged data
            using dg_package_data_projected_tags = tmpl::append<
                grmhd_evolved_vars_tags, fluxes_tags,
                dg_package_data_temporary_tags,
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
                db::get<evolution::dg::Tags::MortarData<3>>(*box),
                Variables<gh_dg_package_field_tags>::
                    number_of_independent_components);

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
        });

    // Now compute the actual time derivatives.
    using gh_variables_tags =
        typename System::gh_system::variables_tag::tags_list;
    using gh_gradient_tags = typename TimeDerivativeTerms::gh_gradient_tags;
    using gh_temporary_tags = typename TimeDerivativeTerms::gh_temp_tags;
    using gh_extra_tags =
        tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                   gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                   ::gh::ConstraintDamping::Tags::ConstraintGamma0,
                   ::gh::ConstraintDamping::Tags::ConstraintGamma1,
                   ::gh::ConstraintDamping::Tags::ConstraintGamma2>;
    using grmhd_source_tags =
        tmpl::transform<ValenciaDivClean::ComputeSources::return_tags,
                        tmpl::bind<db::remove_tag_prefix, tmpl::_1>>;
    using grmhd_source_argument_tags =
        ValenciaDivClean::ComputeSources::argument_tags;
    detail::ComputeTimeDerivImpl<gh_variables_tags, gh_temporary_tags,
                                 gh_gradient_tags, gh_extra_tags,
                                 grmhd_evolved_vars_tags, grmhd_source_tags,
                                 grmhd_source_argument_tags>::
        apply(box, inertial_coords,
              cell_centered_logical_to_inertial_inv_jacobian,
              cell_centered_logical_to_grid_inv_jacobian, one_over_delta_xi,
              boundary_corrections, cell_centered_gh_derivs);
  }
};
}  // namespace grmhd::GhValenciaDivClean::subcell
