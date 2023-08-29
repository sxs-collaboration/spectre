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
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
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
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

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
      const Scalar<DataVector>& cell_centered_det_inv_jacobian,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>&
          cell_centered_logical_to_inertial_inv_jacobian,
      const std::array<double, 3>& one_over_delta_xi,
      const std::array<Variables<tmpl::list<GrmhdDtTags...>>, 3>&
          boundary_corrections,
      const Variables<
          db::wrap_tags_in<::Tags::deriv, typename System::gradients_tags,
                           tmpl::size_t<3>, Frame::Inertial>>& gh_derivs) {
    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
    const size_t number_of_points = subcell_mesh.number_of_grid_points();
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

    // Velocity of the moving mesh, if applicable. We project the value
    // stored on the DG grid onto the subcell grid.
    const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(*box);
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        mesh_velocity_dg = db::get<domain::Tags::MeshVelocity<3>>(*box);
    const std::optional<Scalar<DataVector>>& div_mesh_velocity_dg =
        db::get<domain::Tags::DivMeshVelocity>(*box);
    std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
        mesh_velocity_subcell = {};
    if (mesh_velocity_dg.has_value()) {
      mesh_velocity_subcell = tnsr::I<DataVector, 3, Frame::Inertial>{
          subcell_mesh.number_of_grid_points()};
      for (size_t i = 0; i < 3; i++) {
        mesh_velocity_subcell.value().get(i) =
            evolution::dg::subcell::fd::project(mesh_velocity_dg.value().get(i),
                                                dg_mesh,
                                                subcell_mesh.extents());
      }
    }

    gh::TimeDerivative<3_st>::apply(
        get<::Tags::dt<GhDtTags>>(dt_vars_ptr)...,
        get<GhTemporaries>(temp_tags_ptr)...,
        get<::Tags::deriv<GhGradientTags, tmpl::size_t<3>, Frame::Inertial>>(
            gh_derivs)...,
        get<GhExtraTags>(evolved_vars, temp_tags)...,

        db::get<::gh::gauges::Tags::GaugeCondition>(*box),
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box), time,
        inertial_coords, cell_centered_logical_to_inertial_inv_jacobian,
        mesh_velocity_subcell);
    if (get<gh::gauges::Tags::GaugeCondition>(*box).is_harmonic()) {
      get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(*temp_tags_ptr)) =
          sqrt(
              get(get<gr::Tags::DetSpatialMetric<DataVector>>(*temp_tags_ptr)));
    }

    // Add source terms from moving mesh
    if (mesh_velocity_dg.has_value()) {
      tmpl::for_each<tmpl::list<GhDtTags...>>([&dt_vars_ptr,
                                               &mesh_velocity_subcell,
                                               &gh_derivs](
                                                  auto evolved_var_tag_v) {
        using evolved_var_tag = tmpl::type_from<decltype(evolved_var_tag_v)>;
        using dt_tag = ::Tags::dt<evolved_var_tag>;
        using grad_tag =
            ::Tags::deriv<evolved_var_tag, tmpl::size_t<3>, Frame::Inertial>;
        // Flux and gradients use the same indexing conventions,
        // replacing the direction of the face with the direction
        // of the derivative.
        using FluxTensor = typename grad_tag::type;
        auto& dt_var = get<dt_tag>(*dt_vars_ptr);
        const auto& grad_var = get<grad_tag>(gh_derivs);
        for (size_t i = 0; i < dt_var.size(); ++i) {
          const auto tensor_index = dt_var.get_tensor_index(i);
          for (size_t j = 0; j < 3; j++) {
            const auto grad_index =
                FluxTensor::get_storage_index(prepend(tensor_index, j));
            // Add (mesh_velocity)^j grad_j (var[i])
            dt_var[i] +=
                mesh_velocity_subcell.value().get(j) * grad_var[grad_index];
          }
        }
      });
    }

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

    // Zero GRMHD tags that don't have sources.
    tmpl::for_each<tmpl::list<GrmhdDtTags...>>([&dt_vars_ptr](
                                                   auto evolved_var_tag_v) {
      using evolved_var_tag = tmpl::type_from<decltype(evolved_var_tag_v)>;
      using dt_tag = ::Tags::dt<evolved_var_tag>;
      auto& dt_var = get<dt_tag>(*dt_vars_ptr);
      for (size_t i = 0; i < dt_var.size(); ++i) {
        if constexpr (not tmpl::list_contains_v<tmpl::list<GrmhdSourceTags...>,
                                                evolved_var_tag>) {
          // Zero the GRMHD dt(u) for variables that do not have a source term .
          // This is necessary to avoid `+=` to a `NaN` (debug mode) or random
          // garbage (release mode) when adding to dt_var below.
          dt_var[i] = 0.0;
        }
      }
    });
    // Correction to source terms due to moving mesh
    if (div_mesh_velocity_dg.has_value()) {
      const DataVector div_mesh_velocity_subcell =
          evolution::dg::subcell::fd::project(
              div_mesh_velocity_dg.value().get(), dg_mesh,
              subcell_mesh.extents());
      tmpl::for_each<tmpl::list<GrmhdDtTags...>>(
          [&dt_vars_ptr, &div_mesh_velocity_subcell,
           &evolved_vars](auto evolved_var_tag_v) {
            using evolved_var_tag =
                tmpl::type_from<decltype(evolved_var_tag_v)>;
            using dt_tag = ::Tags::dt<evolved_var_tag>;
            auto& dt_var = get<dt_tag>(*dt_vars_ptr);
            const auto& evolved_var = get<evolved_var_tag>(evolved_vars);
            for (size_t i = 0; i < dt_var.size(); ++i) {
              dt_var[i] -= div_mesh_velocity_subcell * evolved_var[i];
            }
          });
    }

    const tnsr::ii<DataVector, 3> spatial_metric{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        make_const_view(
            make_not_null(&spatial_metric.get(i, j)),
            get<gr::Tags::SpacetimeMetric<DataVector, 3>>(evolved_vars)
                .get(i + 1, j + 1),
            0, number_of_points);
      }
    }

    tenex::evaluate<ti::i>(get<hydro::Tags::SpatialVelocityOneForm<
                               DataVector, 3, Frame::Inertial>>(temp_tags_ptr),
                           get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                               primitive_vars)(ti::J) *
                               spatial_metric(ti::i, ti::j));

    tenex::evaluate<ti::i>(
        get<hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Inertial>>(
            temp_tags_ptr),
        get<hydro::Tags::MagneticField<DataVector, 3>>(primitive_vars)(ti::J) *
            spatial_metric(ti::i, ti::j));

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
      const double inverse_delta = gsl::at(one_over_delta_xi, dim);
      EXPAND_PACK_LEFT_TO_RIGHT([&dt_vars_ptr, &boundary_correction_in_axis,
                                 &cell_centered_det_inv_jacobian, dim,
                                 inverse_delta, &subcell_mesh]() {
        auto& dt_var = *get<::Tags::dt<GrmhdDtTags>>(dt_vars_ptr);
        const auto& var_correction =
            get<GrmhdDtTags>(boundary_correction_in_axis);
        for (size_t i = 0; i < dt_var.size(); ++i) {
          evolution::dg::subcell::add_cartesian_flux_divergence(
              make_not_null(&dt_var[i]), inverse_delta,
              get(cell_centered_det_inv_jacobian), var_correction[i],
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
 */
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
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

    const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(*box);
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
    const auto& cell_centered_logical_to_inertial_inv_jacobian = db::get<
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(
        *box);
    const auto& inertial_coords =
        db::get<evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>>(
            *box);

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
    std::optional<std::array<gsl::span<std::uint8_t>, 3>>
        reconstruction_order{};

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

    // Velocity of the moving mesh on the dg grid, if applicable.
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        mesh_velocity_dg = db::get<domain::Tags::MeshVelocity<3>>(*box);
    // Inverse jacobian, to be projected on faces
    const auto& inv_jacobian_dg =
        db::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(*box);
    const auto& det_inv_jacobian_dg = db::get<
        domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
        *box);

    // GH+GRMHD is a bit different.
    // 1. Compute GH time derivative, since this will also give us lapse, shift,
    //    etc. that we need to reconstruct.
    // 2. Compute d_t Pi_{ab} source terms from MHD (or do we wait until post
    //    MHD source terms?)
    // 3. Reconstruct MHD+spacetime vars to interfaces
    // 4. Compute MHD time derivatives.
    //
    // Compute FD GH derivatives with neighbor data
    // Use highest possible FD order for number of GZ, 2 * (ghost_zone_size)
    const auto& evolved_vars = db::get<evolved_vars_tag>(*box);
    Variables<db::wrap_tags_in<::Tags::deriv, gradients_tags, tmpl::size_t<3>,
                               Frame::Inertial>>
        cell_centered_gh_derivs{num_pts};
    grmhd::GhValenciaDivClean::fd::spacetime_derivatives(
        make_not_null(&cell_centered_gh_derivs), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            *box),
        recons.ghost_zone_size()*2, subcell_mesh,
        cell_centered_logical_to_inertial_inv_jacobian);

    // Now package the data and compute the correction
    //
    // Note: Assumes a the GH and GRMHD corrections can be invoked separately.
    // This is reasonable since the systems are a tensor product system.
    const auto& base_boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections = typename std::decay_t<
        decltype(base_boundary_correction)>::creatable_classes;
    std::array<Variables<grmhd_evolved_vars_tags>, 3> boundary_corrections{};
    call_with_dynamic_type<void, derived_boundary_corrections>(
        &base_boundary_correction, [&](const auto* gh_grmhd_correction) {
          // Need the GH packaged tags to avoid projecting them.
          using gh_dg_package_field_tags = typename std::decay_t<
              decltype(gh_grmhd_correction
                           ->gh_correction())>::dg_package_field_tags;
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
              &recons, [&box, &package_data_argvars_lower_face,
                        &package_data_argvars_upper_face,
                        &reconstruction_order](const auto& reconstructor) {
                using ReconstructorType =
                    std::decay_t<decltype(*reconstructor)>;
                db::apply<
                    typename ReconstructorType::reconstruction_argument_tags>(
                    [&package_data_argvars_lower_face,
                     &package_data_argvars_upper_face, &reconstructor,
                     &reconstruction_order](const auto&... args) {
                      if constexpr (ReconstructorType::use_adaptive_order) {
                        reconstructor->reconstruct(
                            make_not_null(&package_data_argvars_lower_face),
                            make_not_null(&package_data_argvars_upper_face),
                            make_not_null(&reconstruction_order), args...);
                      } else {
                        (void)reconstruction_order;
                        reconstructor->reconstruct(
                            make_not_null(&package_data_argvars_lower_face),
                            make_not_null(&package_data_argvars_upper_face),
                            args...);
                      }
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

            // Build extents of mesh shifted by half a grid cell in direction i
            const unsigned long& num_subcells_1d = subcell_mesh.extents(0);
            Index<3> face_mesh_extents(std::array<size_t, 3>{
                num_subcells_1d, num_subcells_1d, num_subcells_1d});
            face_mesh_extents[i] = num_subcells_1d + 1;
            // Add moving mesh corrections to the fluxes, if needed
            std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
                mesh_velocity_on_face = {};
            if (mesh_velocity_dg.has_value()) {
              // Project mesh velocity on face mesh.
              // Can we get away with only doing the normal component? It
              // is also used in the packaged data...
              mesh_velocity_on_face = tnsr::I<DataVector, 3, Frame::Inertial>{
                  reconstructed_num_pts};
              for (size_t j = 0; j < 3; j++) {
                // j^th component of the velocity on the i^th directed face
                mesh_velocity_on_face.value().get(j) =
                    evolution::dg::subcell::fd::project_to_face(
                        mesh_velocity_dg.value().get(j), dg_mesh,
                        face_mesh_extents, i);
              }

              tmpl::for_each<grmhd_evolved_vars_tags>(
                  [&vars_upper_face, &vars_lower_face,
                   &mesh_velocity_on_face](auto tag_v) {
                    using tag = tmpl::type_from<decltype(tag_v)>;
                    using flux_tag =
                        ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
                    using FluxTensor = typename flux_tag::type;
                    const auto& var_upper = get<tag>(vars_upper_face);
                    const auto& var_lower = get<tag>(vars_lower_face);
                    auto& flux_upper = get<flux_tag>(vars_upper_face);
                    auto& flux_lower = get<flux_tag>(vars_lower_face);
                    for (size_t storage_index = 0;
                         storage_index < var_upper.size(); ++storage_index) {
                      const auto tensor_index =
                          var_upper.get_tensor_index(storage_index);
                      for (size_t j = 0; j < 3; j++) {
                        const auto flux_storage_index =
                            FluxTensor::get_storage_index(
                                prepend(tensor_index, j));
                        flux_upper[flux_storage_index] -=
                            mesh_velocity_on_face.value().get(j) *
                            var_upper[storage_index];
                        flux_lower[flux_storage_index] -=
                            mesh_velocity_on_face.value().get(j) *
                            var_lower[storage_index];
                      }
                    }
                  });
            }

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
            //
            // The unnormalized normal vector is
            // n_j = d \xi^{\hat i}/dx^j
            // with "i" the current face.
            tnsr::i<DataVector, 3, Frame::Inertial> lower_outward_conormal{
                reconstructed_num_pts, 0.0};
            for (size_t j = 0; j < 3; j++) {
              lower_outward_conormal.get(j) =
                  evolution::dg::subcell::fd::project_to_face(
                      inv_jacobian_dg.get(i, j), dg_mesh, face_mesh_extents, i);
            }
            const auto det_inv_jacobian_face =
                evolution::dg::subcell::fd::project_to_face(
                    get(det_inv_jacobian_dg), dg_mesh, face_mesh_extents, i);

            const Scalar<DataVector> normalization{sqrt(get(
                dot_product(lower_outward_conormal, lower_outward_conormal,
                            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                                vars_upper_face))))};
            for (size_t j = 0; j < 3; j++) {
              lower_outward_conormal.get(j) =
                  lower_outward_conormal.get(j) / get(normalization);
            }

            tnsr::i<DataVector, 3, Frame::Inertial> upper_outward_conormal{
                reconstructed_num_pts, 0.0};
            for (size_t j = 0; j < 3; j++) {
              upper_outward_conormal.get(j) = -lower_outward_conormal.get(j);
            }
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
                vars_upper_face, upper_outward_conormal, mesh_velocity_on_face,
                *box, typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});

            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&lower_packaged_data),
                dynamic_cast<const DerivedCorrection&>(boundary_correction),
                vars_lower_face, lower_outward_conormal, mesh_velocity_on_face,
                *box, typename DerivedCorrection::dg_package_data_volume_tags{},
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
            // Also multiply by determinant of Jacobian, following Eq.(34)
            // of 2109.11645
            gsl::at(boundary_corrections, i) *= 1.0 / det_inv_jacobian_face;
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
              db::get<evolution::dg::subcell::fd::Tags::
                          DetInverseJacobianLogicalToInertial>(*box),
              cell_centered_logical_to_inertial_inv_jacobian, one_over_delta_xi,
              boundary_corrections, cell_centered_gh_derivs);
    evolution::dg::subcell::store_reconstruction_order_in_databox(
        box, reconstruction_order);
  }
};
}  // namespace grmhd::GhValenciaDivClean::subcell
