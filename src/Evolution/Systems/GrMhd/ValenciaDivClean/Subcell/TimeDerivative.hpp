// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/HighOrderFluxCorrection.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
 */
struct TimeDerivative {
  template <typename DbTagsList>
  static void apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>&
          cell_centered_logical_to_grid_inv_jacobian,
      const Scalar<DataVector>& /*cell_centered_det_inv_jacobian*/) {
    using evolved_vars_tag = typename System::variables_tag;
    using evolved_vars_tags = typename evolved_vars_tag::tags_list;
    using prim_tags = typename System::primitive_variables_tag::tags_list;
    using recons_prim_tags = tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<3>, Frame::Inertial>;

    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
    const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(*box);
    ASSERT(
        subcell_mesh == Mesh<3>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                                subcell_mesh.quadrature(0)),
        "The subcell/FD mesh must be isotropic for the FD time derivative but "
        "got "
            << subcell_mesh);
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

    // Velocity of the moving mesh on the DG grid, if applicable.
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        mesh_velocity = db::get<domain::Tags::MeshVelocity<3>>(*box);
    const std::optional<Scalar<DataVector>>& div_mesh_velocity =
        db::get<domain::Tags::DivMeshVelocity>(*box);

    const grmhd::ValenciaDivClean::fd::Reconstructor& recons =
        db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(*box);

    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);
    const auto fd_derivative_order =
        db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(*box)
            .finite_difference_derivative_order();
    std::optional<std::array<std::vector<std::uint8_t>, 3>>
        reconstruction_order_data{};
    std::optional<std::array<gsl::span<std::uint8_t>, 3>>
        reconstruction_order{};
    if (static_cast<int>(fd_derivative_order) < 0) {
      reconstruction_order_data = make_array<3>(std::vector<std::uint8_t>(
          (subcell_mesh.extents(0) + 2) * subcell_mesh.extents(1) *
              subcell_mesh.extents(2),
          std::numeric_limits<std::uint8_t>::max()));
      reconstruction_order = std::array<gsl::span<std::uint8_t>, 3>{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(reconstruction_order.value(), i) = gsl::make_span(
            gsl::at(reconstruction_order_data.value(), i).data(),
            gsl::at(reconstruction_order_data.value(), i).size());
      }
    }

    const bool element_is_interior = element.external_boundaries().empty();
    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::SubcellOptions::subcell_enabled_at_external_boundary;

    ASSERT(element_is_interior or subcell_enabled_at_external_boundary,
           "Subcell time derivative is called at a boundary element while "
           "using subcell is disabled at external boundaries."
           "ElementID "
               << element.id());

    // Now package the data and compute the correction
    const auto& boundary_correction =
        db::get<evolution::Tags::BoundaryCorrection<System>>(*box);
    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;
    std::array<Variables<evolved_vars_tags>, 3> boundary_corrections{};

    // If the element has external boundaries and subcell is enabled for
    // boundary elements, compute FD ghost data with a given boundary condition.
    if constexpr (subcell_enabled_at_external_boundary) {
      if (not element.external_boundaries().empty()) {
        fd::BoundaryConditionGhostData::apply(box, element, recons);
      }
    }

    call_with_dynamic_type<void, derived_boundary_corrections>(
        &boundary_correction, [&](const auto* derived_correction) {
          using DerivedCorrection = std::decay_t<decltype(*derived_correction)>;
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
          // Copy over the face values of the metric quantities.
          using spacetime_vars_to_copy =
              tmpl::list<gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>>;
          tmpl::for_each<spacetime_vars_to_copy>(
              [&package_data_argvars_lower_face,
               &package_data_argvars_upper_face,
               &spacetime_vars_on_face =
                   db::get<evolution::dg::subcell::Tags::OnSubcellFaces<
                       typename System::flux_spacetime_variables_tag, 3>>(
                       *box)](auto tag_v) {
                using tag = tmpl::type_from<decltype(tag_v)>;
                for (size_t d = 0; d < 3; ++d) {
                  get<tag>(gsl::at(package_data_argvars_lower_face, d)) =
                      get<tag>(gsl::at(spacetime_vars_on_face, d));
                  get<tag>(gsl::at(package_data_argvars_upper_face, d)) =
                      get<tag>(gsl::at(spacetime_vars_on_face, d));
                }
              });

          // Reconstruct data to the face
          call_with_dynamic_type<void, typename grmhd::ValenciaDivClean::fd::
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

            // Add moving mesh corrections to the fluxes, if needed
            if (mesh_velocity.has_value()) {
              // Build extents of mesh shifted by half a grid cell in direction
              // i
              const unsigned long& num_subcells_1d = subcell_mesh.extents(0);
              Index<3> face_mesh_extents(std::array<size_t, 3>{
                  num_subcells_1d, num_subcells_1d, num_subcells_1d});
              face_mesh_extents[i] = num_subcells_1d + 1;
              // Project mesh velocity on face mesh. We only need the component
              // orthogonal to the face.
              const DataVector& mesh_velocity_on_face =
                  evolution::dg::subcell::fd::project_to_face(
                      mesh_velocity.value().get(i), dg_mesh, face_mesh_extents,
                      i);

              tmpl::for_each<evolved_vars_tags>([&vars_upper_face,
                                                 &vars_lower_face,
                                                 &mesh_velocity_on_face,
                                                 &i](auto tag_v) {
                using tag = tmpl::type_from<decltype(tag_v)>;
                using flux_tag =
                    ::Tags::Flux<tag, tmpl::size_t<3>, Frame::Inertial>;
                using FluxTensor = typename flux_tag::type;
                const auto& var_upper = get<tag>(vars_upper_face);
                const auto& var_lower = get<tag>(vars_lower_face);
                auto& flux_upper = get<flux_tag>(vars_upper_face);
                auto& flux_lower = get<flux_tag>(vars_lower_face);
                for (size_t storage_index = 0; storage_index < var_upper.size();
                     ++storage_index) {
                  const auto tensor_index =
                      var_upper.get_tensor_index(storage_index);
                  const auto flux_storage_index =
                      FluxTensor::get_storage_index(prepend(tensor_index, i));
                  flux_upper[flux_storage_index] -=
                      mesh_velocity_on_face * var_upper[storage_index];
                  flux_lower[flux_storage_index] -=
                      mesh_velocity_on_face * var_lower[storage_index];
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
                evolved_vars_tags, fluxes_tags, dg_package_data_temporary_tags,
                typename DerivedCorrection::dg_package_data_primitive_tags>;
            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&upper_packaged_data), *derived_correction,
                vars_upper_face, upper_outward_conormal, {std::nullopt}, *box,
                typename DerivedCorrection::dg_package_data_volume_tags{},
                dg_package_data_projected_tags{});

            evolution::dg::Actions::detail::dg_package_data<System>(
                make_not_null(&lower_packaged_data), *derived_correction,
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
                db::get<evolution::dg::Tags::MortarData<3>>(*box), 0);

            // Compute the corrections on the faces. We only need to
            // compute this once because we can just flip the normal
            // vectors then
            gsl::at(boundary_corrections, i).initialize(reconstructed_num_pts);
            evolution::dg::subcell::compute_boundary_terms(
                make_not_null(&gsl::at(boundary_corrections, i)),
                *derived_correction, upper_packaged_data, lower_packaged_data);
            // We need to multiply by the normal vector normalization
            gsl::at(boundary_corrections, i) *= get(normalization);
          }
        });

    // Now compute the actual time derivatives.
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
    const gsl::not_null<typename dt_variables_tag::type*> dt_vars_ptr =
        db::mutate<dt_variables_tag>(
            [](const auto local_dt_vars_ptr) { return local_dt_vars_ptr; },
            box);
    dt_vars_ptr->initialize(subcell_mesh.number_of_grid_points());

    using grmhd_source_tags =
        tmpl::transform<ValenciaDivClean::ComputeSources::return_tags,
                        tmpl::bind<db::remove_tag_prefix, tmpl::_1>>;
    sources_impl(
        dt_vars_ptr, *box, grmhd_source_tags{},
        typename grmhd::ValenciaDivClean::ComputeSources::argument_tags{});

    // Zero GRMHD tags that don't have sources.
    tmpl::for_each<typename variables_tag::tags_list>(
        [&dt_vars_ptr](auto evolved_var_tag_v) {
          using evolved_var_tag = tmpl::type_from<decltype(evolved_var_tag_v)>;
          using dt_tag = ::Tags::dt<evolved_var_tag>;
          auto& dt_var = get<dt_tag>(*dt_vars_ptr);
          for (size_t i = 0; i < dt_var.size(); ++i) {
            if constexpr (not tmpl::list_contains_v<grmhd_source_tags,
                                                    evolved_var_tag>) {
              dt_var[i] = 0.0;
            }
          }
        });

    // Correction to source terms due to moving mesh
    if (div_mesh_velocity.has_value()) {
      const DataVector div_mesh_velocity_subcell =
          evolution::dg::subcell::fd::project(div_mesh_velocity.value().get(),
                                              dg_mesh, subcell_mesh.extents());
      const auto& evolved_vars = db::get<evolved_vars_tag>(*box);

      tmpl::for_each<typename variables_tag::tags_list>(
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

    std::optional<std::array<Variables<evolved_vars_tags>, 3>>
        high_order_corrections{};
    ::fd::cartesian_high_order_flux_corrections(
        make_not_null(&high_order_corrections),

        db::get<evolution::dg::subcell::Tags::CellCenteredFlux<
            evolved_vars_tags, 3>>(*box),
        boundary_corrections, fd_derivative_order,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            *box),
        subcell_mesh, recons.ghost_zone_size(),
        reconstruction_order.value_or(
            std::array<gsl::span<std::uint8_t>, 3>{}));

    for (size_t dim = 0; dim < 3; ++dim) {
      const auto& boundary_correction_in_axis =
          high_order_corrections.has_value()
              ? gsl::at(high_order_corrections.value(), dim)
              : gsl::at(boundary_corrections, dim);
      const auto& component_inverse_jacobian =
          cell_centered_logical_to_grid_inv_jacobian.get(dim, dim);
      const double inverse_delta = gsl::at(one_over_delta_xi, dim);
      tmpl::for_each<typename variables_tag::tags_list>(
          [&dt_vars_ptr, &boundary_correction_in_axis,
           &component_inverse_jacobian, dim, inverse_delta,
           &subcell_mesh](auto evolved_var_tag_v) {
            using evolved_var_tag =
                tmpl::type_from<decltype(evolved_var_tag_v)>;
            using dt_tag = ::Tags::dt<evolved_var_tag>;
            auto& dt_var = get<dt_tag>(*dt_vars_ptr);
            const auto& var_correction =
                get<evolved_var_tag>(boundary_correction_in_axis);
            for (size_t i = 0; i < dt_var.size(); ++i) {
              evolution::dg::subcell::add_cartesian_flux_divergence(
                  make_not_null(&dt_var[i]), inverse_delta,
                  component_inverse_jacobian, var_correction[i],
                  subcell_mesh.extents(), dim);
            }
          });
    }

    evolution::dg::subcell::store_reconstruction_order_in_databox(
        box, reconstruction_order);
  }

 private:
  template <typename DtVarsList, typename DbTagsList, typename... SourcedTags,
            typename... ArgsTags>
  static void sources_impl(
      const gsl::not_null<Variables<DtVarsList>*> dt_vars_ptr,
      const db::DataBox<DbTagsList>& box, tmpl::list<SourcedTags...> /*meta*/,
      tmpl::list<ArgsTags...> /*meta*/) {
    grmhd::ValenciaDivClean::ComputeSources::apply(
        get<::Tags::dt<SourcedTags>>(dt_vars_ptr)..., get<ArgsTags>(box)...);
  }
};
}  // namespace grmhd::ValenciaDivClean::subcell
