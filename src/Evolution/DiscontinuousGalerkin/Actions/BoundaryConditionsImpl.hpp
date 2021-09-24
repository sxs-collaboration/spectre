// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/InterpolateFromBoundary.hpp"
#include "Evolution/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions::detail {
template <typename BoundaryConditionHelper, typename AllTagsOnFaceList,
          typename... TagsFromFace, typename... VolumeArgs>
std::optional<std::string> apply_boundary_condition_impl(
    BoundaryConditionHelper& boundary_condition_helper,
    const Variables<AllTagsOnFaceList>& fields_on_interior_face,
    tmpl::list<TagsFromFace...> /*meta*/,
    const VolumeArgs&... volume_args) noexcept {
  return boundary_condition_helper(
      get<TagsFromFace>(fields_on_interior_face)..., volume_args...);
}

template <typename System, size_t Dim, typename DbTagsList,
          typename BoundaryCorrection, typename BoundaryCondition,
          typename... EvolvedVariablesTags, typename... PackageDataVolumeTags,
          typename... BoundaryConditionVolumeTags, typename... PackageFieldTags,
          typename... BoundaryCorrectionPackagedDataInputTags>
void apply_boundary_condition_on_face(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    [[maybe_unused]] const BoundaryCorrection& boundary_correction,
    const BoundaryCondition& boundary_condition,
    const Direction<Dim>& direction,
    [[maybe_unused]] const Variables<tmpl::list<EvolvedVariablesTags...>>&
        volume_evolved_vars,
    [[maybe_unused]] const Variables<
        db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                         tmpl::size_t<Dim>, Frame::Inertial>>& volume_fluxes,
    [[maybe_unused]] const Variables<
        db::wrap_tags_in<::Tags::deriv, typename System::gradient_variables,
                         tmpl::size_t<Dim>, Frame::Inertial>>& partial_derivs,
    [[maybe_unused]] const Variables<
        typename System::compute_volume_time_derivative_terms::temporary_tags>&
        volume_temporaries,
    [[maybe_unused]] const Variables<
        detail::get_primitive_vars_tags_from_system<System>>* const
        volume_primitive_variables,
    [[maybe_unused]] const ::dg::Formulation dg_formulation,
    const Mesh<Dim>& volume_mesh, [[maybe_unused]] const Element<Dim>& element,
    [[maybe_unused]] const ::ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
        moving_mesh_map,
    [[maybe_unused]] const double time,
    [[maybe_unused]] const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::optional<tnsr::I<DataVector, Dim>>& volume_mesh_velocity,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        volume_inverse_jacobian,
    [[maybe_unused]] const Scalar<DataVector>& volume_det_inv_jacobian,
    tmpl::list<PackageDataVolumeTags...> /*meta*/,
    tmpl::list<PackageFieldTags...> /*meta*/,
    tmpl::list<BoundaryCorrectionPackagedDataInputTags...> /*meta*/,
    tmpl::list<BoundaryConditionVolumeTags...> /*meta*/) noexcept {
  using variables_tag = typename System::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using flux_variables = typename System::flux_variables;
  using dt_variables_tags = db::wrap_tags_in<::Tags::dt, variables_tags>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  const Mesh<Dim - 1> face_mesh = volume_mesh.slice_away(direction.dimension());
  const size_t number_of_points_on_face = face_mesh.number_of_grid_points();

  // We figure out all the tags we need to project from the interior, both for
  // the boundary condition computation and for the boundary correction. We do
  // this by:
  // 1. get all interior tags for the boundary condition
  // 2. get all interior tags for the boundary correction (if ghost condition)
  // 3. combine these lists
  // 4. project from the interior
  //
  // Note: we only need to consider the boundary correction tags if a ghost
  // boundary condition is imposed.

  constexpr bool uses_ghost_condition =
      BoundaryCondition::bc_type ==
          evolution::BoundaryConditions::Type::Ghost or
      BoundaryCondition::bc_type ==
          evolution::BoundaryConditions::Type::GhostAndTimeDerivative;
  constexpr bool uses_time_derivative_condition =
      BoundaryCondition::bc_type ==
          evolution::BoundaryConditions::Type::TimeDerivative or
      BoundaryCondition::bc_type ==
          evolution::BoundaryConditions::Type::GhostAndTimeDerivative;
  constexpr bool needs_coordinates = tmpl::list_contains_v<
      typename BoundaryCondition::dg_interior_temporary_tags,
      ::domain::Tags::Coordinates<Dim, Frame::Inertial>>;

  // List that holds the inverse spatial metric if it's needed
  using inverse_spatial_metric_list =
      detail::inverse_spatial_metric_tag<System>;
  constexpr bool has_inv_spatial_metric =
      detail::has_inverse_spatial_metric_tag_v<System>;

  // Set up tags for boundary conditions
  using bcondition_interior_temp_tags =
      typename BoundaryCondition::dg_interior_temporary_tags;
  using bcondition_interior_prim_tags =
      detail::boundary_condition_primitive_tags<
          System::has_primitive_and_conservative_vars, BoundaryCondition>;
  using bcondition_interior_evolved_vars_tags =
      typename BoundaryCondition::dg_interior_evolved_variables_tags;
  using bcondition_interior_dt_evolved_vars_tags =
      detail::get_dt_vars_from_boundary_condition<BoundaryCondition>;
  using bcondition_interior_deriv_evolved_vars_tags =
      detail::get_deriv_vars_from_boundary_condition<BoundaryCondition>;
  using bcondition_interior_tags = tmpl::append<
      tmpl::conditional_t<has_inv_spatial_metric,
                          tmpl::list<detail::NormalVector<Dim>>, tmpl::list<>>,
      bcondition_interior_evolved_vars_tags, bcondition_interior_prim_tags,
      bcondition_interior_temp_tags, bcondition_interior_dt_evolved_vars_tags,
      bcondition_interior_deriv_evolved_vars_tags>;

  // Set up tags for boundary correction
  using correction_temp_tags = tmpl::conditional_t<
      uses_ghost_condition,
      typename BoundaryCorrection::dg_package_data_temporary_tags,
      tmpl::list<>>;
  using correction_prim_tags = tmpl::conditional_t<
      uses_ghost_condition,
      detail::boundary_correction_primitive_tags<
          System::has_primitive_and_conservative_vars, BoundaryCorrection>,
      tmpl::list<>>;
  using correction_evolved_vars_tags =
      tmpl::conditional_t<uses_ghost_condition,
                          typename System::variables_tag::tags_list,
                          tmpl::list<>>;

  // Now combine the tags lists for each type of tag. These are all the tags
  // we need to project from the interior, excluding the inverse spatial
  // metric. They are the input to `dg_package_data` in the boundary
  // correction.
  using interior_temp_tags = tmpl::remove_duplicates<
      tmpl::append<bcondition_interior_temp_tags, correction_temp_tags>>;
  using interior_prim_tags = tmpl::remove_duplicates<
      tmpl::append<bcondition_interior_prim_tags, correction_prim_tags>>;
  using interior_evolved_vars_tags = tmpl::remove_duplicates<tmpl::append<
      correction_evolved_vars_tags, bcondition_interior_evolved_vars_tags>>;

  // List tags on the interior of the face. We list the exterior side
  // separately in the `else` branch of the if-constexpr where we actually use
  // the exterior fields.
  using fluxes_tags =
      tmpl::conditional_t<uses_ghost_condition,
                          db::wrap_tags_in<::Tags::Flux, flux_variables,
                                           tmpl::size_t<Dim>, Frame::Inertial>,
                          tmpl::list<>>;
  using tags_on_interior_face = tmpl::remove_duplicates<tmpl::append<
      fluxes_tags, interior_temp_tags, interior_prim_tags,
      interior_evolved_vars_tags, bcondition_interior_dt_evolved_vars_tags,
      bcondition_interior_deriv_evolved_vars_tags, inverse_spatial_metric_list,
      tmpl::list<detail::OneOverNormalVectorMagnitude,
                 detail::NormalVector<Dim>>>>;

  Variables<tags_on_interior_face> interior_face_fields{
      number_of_points_on_face};

  // Perform projection into `interior_face_fields`. This also covers all the
  // fields for the exterior except for the time derivatives that might be
  // needed for Bjorhus/TimeDerivative boundary conditions.
  //
  // Note on the ordering of the data to project: if we are using a ghost
  // boundary condition with a boundary correction, then we know that all the
  // evolved variables are needed, whereas when using Outflow or Bjorhus
  // boundary conditions none of the evolved variables might be needed (or only
  // some subset). Also, the way the typelist is assembled, the evolved vars are
  // guaranteed to be contiguous, but only if we are doing a ghost boundary
  // condition.
  if constexpr (uses_ghost_condition) {
    project_contiguous_data_to_boundary(make_not_null(&interior_face_fields),
                                        volume_evolved_vars, volume_mesh,
                                        direction);
  } else {
    project_tensors_to_boundary<interior_evolved_vars_tags>(
        make_not_null(&interior_face_fields), volume_evolved_vars, volume_mesh,
        direction);
  }
  if constexpr (tmpl::size<fluxes_tags>::value != 0) {
    project_contiguous_data_to_boundary(make_not_null(&interior_face_fields),
                                        volume_fluxes, volume_mesh, direction);
  } else {
    (void)volume_fluxes;
  }
  using temp_tags_no_coordinates =
      tmpl::remove<interior_temp_tags,
                   domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  if constexpr (tmpl::size<tmpl::append<
                    temp_tags_no_coordinates,
                    detail::inverse_spatial_metric_tag<System>>>::value != 0) {
    project_tensors_to_boundary<tmpl::append<
        temp_tags_no_coordinates, detail::inverse_spatial_metric_tag<System>>>(
        make_not_null(&interior_face_fields), volume_temporaries, volume_mesh,
        direction);
  }
  if constexpr (System::has_primitive_and_conservative_vars and
                tmpl::size<interior_prim_tags>::value != 0) {
    ASSERT(volume_primitive_variables != nullptr,
           "The volume primitive variables are not set even though the "
           "system has primitive variables.");
    project_tensors_to_boundary<interior_prim_tags>(
        make_not_null(&interior_face_fields), *volume_primitive_variables,
        volume_mesh, direction);
  } else {
    (void)volume_primitive_variables;
  }
  if constexpr (tmpl::size<
                    bcondition_interior_deriv_evolved_vars_tags>::value != 0) {
    project_tensors_to_boundary<bcondition_interior_deriv_evolved_vars_tags>(
        make_not_null(&interior_face_fields), partial_derivs, volume_mesh,
        direction);
  }
  if constexpr (tmpl::size<bcondition_interior_dt_evolved_vars_tags>::value !=
                0) {
    project_tensors_to_boundary<bcondition_interior_dt_evolved_vars_tags>(
        make_not_null(&interior_face_fields), db::get<dt_variables_tag>(*box),
        volume_mesh, direction);
  }

  std::optional<tnsr::I<DataVector, Dim>> face_mesh_velocity{};
  if (volume_mesh_velocity.has_value()) {
    face_mesh_velocity = tnsr::I<DataVector, Dim>{number_of_points_on_face};
    project_tensor_to_boundary(make_not_null(&*face_mesh_velocity),
                               *volume_mesh_velocity, volume_mesh, direction);
  }

  // Normalize the normal vectors. We cache the unit normal covector For
  // flat geometry and static meshes.
  const auto normalize_normal_vectors =
      [&direction, mesh_is_moving = not moving_mesh_map.is_identity(),
       number_of_points_on_face, &volume_inverse_jacobian,
       &volume_mesh](const auto normal_covector_magnitude_in_direction_ptr,
                     auto fields_on_face_ptr) noexcept {
        if (auto& normal_covector_quantity =
                *normal_covector_magnitude_in_direction_ptr;
            has_inv_spatial_metric or mesh_is_moving or
            not normal_covector_quantity.has_value()) {
          if (not normal_covector_quantity.has_value()) {
            normal_covector_quantity =
                Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                     evolution::dg::Tags::NormalCovector<Dim>>>{
                    number_of_points_on_face};
          }
          tnsr::i<DataVector, Dim> volume_unnormalized_normal_covector{};

          for (size_t inertial_index = 0; inertial_index < Dim;
               ++inertial_index) {
            volume_unnormalized_normal_covector.get(inertial_index)
                .set_data_ref(
                    const_cast<double*>(  // NOLINT
                        volume_inverse_jacobian
                            .get(direction.dimension(), inertial_index)
                            .data()),
                    volume_mesh.number_of_grid_points());
          }
          project_tensor_to_boundary(
              make_not_null(&get<evolution::dg::Tags::NormalCovector<Dim>>(
                  *normal_covector_quantity)),
              volume_unnormalized_normal_covector, volume_mesh, direction);

          if (const double sign = direction.sign(); sign != 1.0) {
            for (auto& normal_covector_component :
                 get<evolution::dg::Tags::NormalCovector<Dim>>(
                     *normal_covector_quantity)) {
              normal_covector_component *= sign;
            }
          }

          detail::unit_normal_vector_and_covector_and_magnitude_impl<System>(
              make_not_null(&get<evolution::dg::Tags::MagnitudeOfNormal>(
                  *normal_covector_quantity)),
              make_not_null(&get<evolution::dg::Tags::NormalCovector<Dim>>(
                  *normal_covector_quantity)),
              fields_on_face_ptr,
              get<evolution::dg::Tags::NormalCovector<Dim>>(
                  *normal_covector_quantity));
        }
      };
  // Normalize the outward facing normal vector on the interior side
  db::mutate<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(
      box, [&direction, &interior_face_fields, &normalize_normal_vectors](
               const auto normal_covector_and_magnitude_ptr) noexcept {
        normalize_normal_vectors(
            make_not_null(&normal_covector_and_magnitude_ptr->at(direction)),
            make_not_null(&interior_face_fields));
      });

  const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_normal_covector =
      get<evolution::dg::Tags::NormalCovector<Dim>>(
          *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(*box)
               .at(direction));

  if constexpr (needs_coordinates) {
    // Compute the coordinates on the interface
    get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(interior_face_fields) =
        moving_mesh_map(logical_to_grid_map(interface_logical_coordinates(
                            face_mesh, direction)),
                        time, functions_of_time);
  }

  if constexpr (BoundaryCondition::bc_type ==
                evolution::BoundaryConditions::Type::Outflow) {
    // Outflow boundary conditions only check that all characteristic speeds
    // are directed out of the element. If there are any inward directed
    // fields then the boundary condition should error.
    const auto apply_bc = [&boundary_condition, &face_mesh_velocity,
                           &interior_normal_covector](
                              const auto&... face_and_volume_args) noexcept {
      return boundary_condition.dg_outflow(face_mesh_velocity,
                                           interior_normal_covector,
                                           face_and_volume_args...);
    };
    const std::optional<std::string> error_message =
        apply_boundary_condition_impl(
            apply_bc, interior_face_fields, bcondition_interior_tags{},
            db::get<BoundaryConditionVolumeTags>(*box)...);
    if (error_message.has_value()) {
      ERROR(*error_message << "\n\nIn element:" << element.id()
                           << "\nIn direction: " << direction);
    }
    return;
  }

  // We add the time derivative boundary conditions and lift the ghost boundary
  // conditions after both have been computed in case either depends on the
  // time derivatives in the volume projected on to the face.

  Variables<dt_variables_tags> dt_time_derivative_correction{};
  if constexpr (uses_time_derivative_condition) {
    dt_time_derivative_correction.initialize(number_of_points_on_face);
    auto apply_bc = [&boundary_condition, &dt_time_derivative_correction,
                     &face_mesh_velocity, &interior_normal_covector](
                        const auto&... interior_face_and_volume_args) {
      return boundary_condition.dg_time_derivative(
          make_not_null(&get<::Tags::dt<EvolvedVariablesTags>>(
              dt_time_derivative_correction))...,
          face_mesh_velocity, interior_normal_covector,
          interior_face_and_volume_args...);
    };
    const std::optional<std::string> error_message =
        apply_boundary_condition_impl(
            apply_bc, interior_face_fields, bcondition_interior_tags{},
            db::get<BoundaryConditionVolumeTags>(*box)...);
    if (error_message.has_value()) {
      ERROR(*error_message << "\n\nIn element:" << element.id()
                           << "\nIn direction: " << direction);
    }
  } else {
    (void)dt_time_derivative_correction;
  }

  // Now we populate the fields on the exterior side of the face using the
  // boundary condition.
  using tags_on_exterior_face =
      tmpl::append<variables_tags, fluxes_tags, correction_temp_tags,
                   correction_prim_tags, inverse_spatial_metric_list,
                   tmpl::list<detail::OneOverNormalVectorMagnitude,
                              detail::NormalVector<Dim>,
                              evolution::dg::Tags::NormalCovector<Dim>>>;
  Variables<tags_on_exterior_face> exterior_face_fields{
      number_of_points_on_face};

  if constexpr (uses_ghost_condition) {
    using mortar_tags_list = tmpl::list<PackageFieldTags...>;
    using dg_package_data_projected_tags =
        tmpl::append<variables_tags, fluxes_tags, correction_temp_tags,
                     correction_prim_tags>;

    Variables<mortar_tags_list> internal_packaged_data{
        number_of_points_on_face};
    const double max_abs_char_speed_on_face = detail::dg_package_data<System>(
        make_not_null(&internal_packaged_data), boundary_correction,
        interior_face_fields, interior_normal_covector, face_mesh_velocity,
        dg_package_data_projected_tags{},
        db::get<PackageDataVolumeTags>(*box)...);
    (void)max_abs_char_speed_on_face;

    // Notes:
    // - we pass the outward directed normal vector normalized using the
    //   interior variables to the boundary condition. This is because the
    //   boundary condition should only need the normal vector for computing
    //   things like reflecting BCs where the normal component of an interior
    //   quantity is reversed.
    // - if needed, the boundary condition returns the inverse spatial metric on
    //   the exterior side, which is then used to normalize the normal vector on
    //   the exterior side. We need the exterior normal vector for computing
    //   flux terms. The inverse spatial metric on the exterior side can be
    //   equal to the inverse spatial metric on the interior side. This would be
    //   true when, e.g. imposing reflecting boundary conditions.
    // - in addition to the evolved variables and fluxes, the boundary condition
    //   must compute the `dg_packaged_data_temporary_tags` and the primitive
    //   tags that the boundary correction needs.
    // - For systems with constraint damping parameters, the constraint damping
    //   parameters are just copied from the projected values from the interior.
    auto apply_bc = [&boundary_condition, &exterior_face_fields,
                     &face_mesh_velocity, &interior_normal_covector](
                        const auto&... interior_face_and_volume_args) {
      if constexpr (has_inv_spatial_metric) {
        return boundary_condition.dg_ghost(
            make_not_null(&get<BoundaryCorrectionPackagedDataInputTags>(
                exterior_face_fields))...,
            make_not_null(
                &get<tmpl::front<detail::inverse_spatial_metric_tag<System>>>(
                    exterior_face_fields)),
            face_mesh_velocity, interior_normal_covector,
            interior_face_and_volume_args...);
      } else {
        return boundary_condition.dg_ghost(
            make_not_null(&get<BoundaryCorrectionPackagedDataInputTags>(
                exterior_face_fields))...,
            face_mesh_velocity, interior_normal_covector,
            interior_face_and_volume_args...);
      }
    };
    const std::optional<std::string> error_message =
        apply_boundary_condition_impl(
            apply_bc, interior_face_fields, bcondition_interior_tags{},
            db::get<BoundaryConditionVolumeTags>(*box)...);
    if (error_message.has_value()) {
      ERROR(*error_message << "\n\nIn element:" << element.id()
                           << "\nIn direction: " << direction);
    }
    // Subtract mesh velocity from the _exterior_ fluxes
    if (face_mesh_velocity.has_value()) {
      tmpl::for_each<flux_variables>(
          [&face_mesh_velocity, &exterior_face_fields](auto tag_v) noexcept {
            // Modify fluxes for moving mesh
            using var_tag = typename decltype(tag_v)::type;
            using flux_var_tag =
                db::add_tag_prefix<::Tags::Flux, var_tag, tmpl::size_t<Dim>,
                                   Frame::Inertial>;
            auto& flux_var = get<flux_var_tag>(exterior_face_fields);
            const auto& var = get<var_tag>(exterior_face_fields);
            const auto& mesh_velocity = *face_mesh_velocity;
            // Loop over all independent components of flux_var
            for (size_t flux_var_storage_index = 0;
                 flux_var_storage_index < flux_var.size();
                 ++flux_var_storage_index) {
              // Get the flux variable's tensor index, e.g. (i,j) for a F^i of
              // the spatial velocity (or some other spatial tensor).
              const auto flux_var_tensor_index =
                  flux_var.get_tensor_index(flux_var_storage_index);
              // Remove the first index from the flux tensor index, gets back
              // (j)
              const auto var_tensor_index =
                  all_but_specified_element_of(flux_var_tensor_index, 0);
              // Set flux_index to (i)
              const size_t flux_index = gsl::at(flux_var_tensor_index, 0);

              // We now need to index flux(i,j) -= u(j) * v_g(i)
              flux_var[flux_var_storage_index] -=
                  var.get(var_tensor_index) * mesh_velocity.get(flux_index);
            }
          });
    }
    // Now that we have computed the inverse spatial metric on the exterior, we
    // can compute the normalized normal (co)vector on the exterior side. If
    // there is no inverse spatial metric, then we just copy from the interior
    // and reverse the sign.
    for (size_t i = 0; i < Dim; ++i) {
      get<evolution::dg::Tags::NormalCovector<Dim>>(exterior_face_fields)
          .get(i) = -interior_normal_covector.get(i);
    }
    if constexpr (has_inv_spatial_metric) {
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric =
          get<tmpl::front<inverse_spatial_metric_list>>(exterior_face_fields);
      tnsr::i<DataVector, Dim, Frame::Inertial>& exterior_normal_covector =
          get<evolution::dg::Tags::NormalCovector<Dim>>(exterior_face_fields);
      tnsr::I<DataVector, Dim, Frame::Inertial>& exterior_normal_vector =
          get<detail::NormalVector<Dim>>(exterior_face_fields);

      // Since the spatial metric is different on the exterior side of the
      // interface, we need to normalize the direction-reversed interior normal
      // vector using the exterior inverse spatial metric.
      for (size_t i = 0; i < Dim; ++i) {
        exterior_normal_vector.get(i) =
            get<0>(exterior_normal_covector) * inv_spatial_metric.get(i, 0);
        for (size_t j = 1; j < Dim; ++j) {
          exterior_normal_vector.get(i) +=
              exterior_normal_covector.get(j) * inv_spatial_metric.get(i, j);
        }
      }
      // Use detail::OneOverNormalVectorMagnitude as a buffer for the
      // magnitude. We don't need one over the normal magnitude on the
      // exterior side since we aren't lifting there.
      Scalar<DataVector>& magnitude =
          get<detail::OneOverNormalVectorMagnitude>(exterior_face_fields);
      dot_product(make_not_null(&magnitude), exterior_normal_covector,
                  exterior_normal_vector);
      get(magnitude) = sqrt(get(magnitude));
      for (size_t i = 0; i < Dim; ++i) {
        exterior_normal_covector.get(i) /= get(magnitude);
        exterior_normal_vector.get(i) /= get(magnitude);
      }
    }

    // Package the external-side data for the boundary correction
    Variables<mortar_tags_list> external_packaged_data{
        number_of_points_on_face};
    detail::dg_package_data<System>(
        make_not_null(&external_packaged_data), boundary_correction,
        exterior_face_fields,
        get<evolution::dg::Tags::NormalCovector<Dim>>(exterior_face_fields),
        face_mesh_velocity, dg_package_data_projected_tags{},
        db::get<PackageDataVolumeTags>(*box)...);

    Variables<dt_variables_tags> boundary_corrections_on_face{
        number_of_points_on_face};

    // Compute boundary correction
    boundary_correction.dg_boundary_terms(
        make_not_null(&get<::Tags::dt<EvolvedVariablesTags>>(
            boundary_corrections_on_face))...,
        get<PackageFieldTags>(internal_packaged_data)...,
        get<PackageFieldTags>(external_packaged_data)..., dg_formulation);

    // Lift the boundary correction
    const auto& magnitude_of_interior_face_normal =
        get<evolution::dg::Tags::MagnitudeOfNormal>(
            *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(*box)
                 .at(direction));
    if (volume_mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
      // The lift_flux function lifts only on the slice, it does not add
      // the contribution to the volume.
      ::dg::lift_flux(make_not_null(&boundary_corrections_on_face),
                      volume_mesh.extents(direction.dimension()),
                      magnitude_of_interior_face_normal);

      // Add the flux contribution to the volume data
      db::mutate<dt_variables_tag>(
          box, [&direction, &boundary_corrections_on_face,
                &volume_mesh](const auto dt_variables_ptr) noexcept {
            add_slice_to_data(
                dt_variables_ptr, boundary_corrections_on_face,
                volume_mesh.extents(), direction.dimension(),
                index_to_slice_at(volume_mesh.extents(), direction));
          });
    } else {
      // We are using Gauss points.
      //
      // Optimization note: eliminate allocations for volume and face det
      // jacobian. Should probably compute face det inv jacobian, then divide
      // (fewer grid points => fewer FLOPs).
      const DataVector volume_det_jacobian = 1.0 / get(volume_det_inv_jacobian);

      // Project the determinant of the Jacobian to the face. This could
      // be optimized by caching in the time-independent case.
      Scalar<DataVector> face_det_jacobian{face_mesh.number_of_grid_points()};
      const Matrix identity{};
      auto interpolation_matrices = make_array<Dim>(std::cref(identity));
      const std::pair<Matrix, Matrix>& matrices =
          Spectral::boundary_interpolation_matrices(
              volume_mesh.slice_through(direction.dimension()));
      gsl::at(interpolation_matrices, direction.dimension()) =
          direction.side() == Side::Upper ? matrices.second : matrices.first;
      apply_matrices(make_not_null(&get(face_det_jacobian)),
                     interpolation_matrices, volume_det_jacobian,
                     volume_mesh.extents());

      db::mutate<dt_variables_tag>(
          box, [&direction, &boundary_corrections_on_face, &face_det_jacobian,
                &magnitude_of_interior_face_normal, &volume_det_inv_jacobian,
                &volume_mesh](const auto dt_variables_ptr) noexcept {
            evolution::dg::lift_boundary_terms_gauss_points(
                dt_variables_ptr, volume_det_inv_jacobian, volume_mesh,
                direction, boundary_corrections_on_face,
                magnitude_of_interior_face_normal, face_det_jacobian);
          });
    }
  }
  // Add TimeDerivative correction to volume time derivatives.
  if constexpr (uses_time_derivative_condition) {
    if (volume_mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
      db::mutate<dt_variables_tag>(
          box, [&direction, &dt_time_derivative_correction,
                &volume_mesh](const auto dt_variables_ptr) noexcept {
            add_slice_to_data(
                dt_variables_ptr, dt_time_derivative_correction,
                volume_mesh.extents(), direction.dimension(),
                index_to_slice_at(volume_mesh.extents(), direction));
          });
    } else {
      db::mutate<dt_variables_tag>(
          box, [&direction, &dt_time_derivative_correction,
                &volume_mesh](const auto dt_variables_ptr) noexcept {
            interpolate_dt_terms_gauss_points(dt_variables_ptr, volume_mesh,
                                              direction,
                                              dt_time_derivative_correction);
          });
    }
  }
}

/*!
 * \brief Applies the boundary conditions using the `boundary_correction`
 * on all external faces.
 *
 * A `tmpl::for_each` loop along with a `typeid` comparison checks which of the
 * known boundary conditions is being used. Since each direction can have a
 * different boundary condition, we must check each boundary condition in
 * each external direction.
 */
template <typename System, size_t Dim, typename DbTagsList,
          typename BoundaryCorrection>
void apply_boundary_conditions_on_all_external_faces(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const BoundaryCorrection& boundary_correction,
    const Variables<
        typename System::compute_volume_time_derivative_terms::temporary_tags>&
        temporaries,
    const Variables<
        db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                         tmpl::size_t<Dim>, Frame::Inertial>>& volume_fluxes,
    const Variables<
        db::wrap_tags_in<::Tags::deriv, typename System::gradient_variables,
                         tmpl::size_t<Dim>, Frame::Inertial>>& partial_derivs,
    const Variables<detail::get_primitive_vars_tags_from_system<System>>* const
        primitive_vars) noexcept {
  using factory_classes =
      typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
          *box))>::factory_creation::factory_classes;
  using derived_boundary_conditions = tmpl::remove_if<
      tmpl::at<factory_classes, typename System::boundary_conditions_base>,
      tmpl::or_<
          std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>>>;

  using variables_tag = typename System::variables_tag;
  using flux_variables = typename System::flux_variables;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, flux_variables,
                                       tmpl::size_t<Dim>, Frame::Inertial>;

  const Element<Dim>& element = db::get<domain::Tags::Element<Dim>>(*box);
  size_t number_of_boundaries_left = element.external_boundaries().size();

  if (number_of_boundaries_left == 0) {
    return;
  }

  const auto& external_boundary_conditions =
      db::get<domain::Tags::Domain<Dim>>(*box)
          .blocks()[element.id().block_id()]
          .external_boundary_conditions();
  tmpl::for_each<derived_boundary_conditions>(
      [&boundary_correction, &box, &element, &external_boundary_conditions,
       &number_of_boundaries_left, &partial_derivs, &primitive_vars,
       &temporaries,
       &volume_fluxes](auto derived_boundary_condition_v) noexcept {
        using DerivedBoundaryCondition =
            tmpl::type_from<decltype(derived_boundary_condition_v)>;

        if (number_of_boundaries_left == 0) {
          return;
        }

        for (const Direction<Dim>& direction : element.external_boundaries()) {
          const auto& boundary_condition =
              *external_boundary_conditions.at(direction);
          if (typeid(boundary_condition) == typeid(DerivedBoundaryCondition)) {
            detail::apply_boundary_condition_on_face<System>(
                box, boundary_correction,
                dynamic_cast<const DerivedBoundaryCondition&>(
                    boundary_condition),
                direction, db::get<variables_tag>(*box), volume_fluxes,
                partial_derivs, temporaries, primitive_vars,
                db::get<::dg::Tags::Formulation>(*box),
                db::get<::domain::Tags::Mesh<Dim>>(*box),
                db::get<::domain::Tags::Element<Dim>>(*box),
                db::get<::domain::Tags::ElementMap<Dim, Frame::Grid>>(*box),
                db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
                    Dim, Frame::Grid, Frame::Inertial>>(*box),
                db::get<::Tags::Time>(*box),
                db::get<::domain::Tags::FunctionsOfTime>(*box),
                db::get<::domain::Tags::MeshVelocity<Dim>>(*box),
                db::get<::domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                        Frame::Inertial>>(*box),
                db::get<::domain::Tags::DetInvJacobian<Frame::Logical,
                                                       Frame::Inertial>>(*box),
                typename BoundaryCorrection::dg_package_data_volume_tags{},
                typename BoundaryCorrection::dg_package_field_tags{},
                tmpl::append<
                    typename variables_tag::tags_list, fluxes_tags,
                    typename BoundaryCorrection::dg_package_data_temporary_tags,
                    typename detail::get_primitive_vars<
                        System::has_primitive_and_conservative_vars>::
                        template f<BoundaryCorrection>>{},
                typename DerivedBoundaryCondition::dg_gridless_tags{});
            --number_of_boundaries_left;
          }
          if (number_of_boundaries_left == 0) {
            return;
          }
        }
      });
}
}  // namespace evolution::dg::Actions::detail
