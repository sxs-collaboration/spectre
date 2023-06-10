// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions::detail {
template <typename System, size_t Dim, typename BoundaryCorrection,
          typename TemporaryTags, typename... PackageDataVolumeArgs>
void internal_mortar_data_impl(
    const gsl::not_null<
        DirectionMap<Dim, std::optional<Variables<tmpl::list<
                              evolution::dg::Tags::MagnitudeOfNormal,
                              evolution::dg::Tags::NormalCovector<Dim>>>>>*>
        normal_covector_and_magnitude_ptr,
    const gsl::not_null<std::unordered_map<
        std::pair<Direction<Dim>, ElementId<Dim>>,
        evolution::dg::MortarData<Dim>,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
        mortar_data_ptr,
    const gsl::not_null<gsl::span<double>*> face_temporaries,
    const gsl::not_null<gsl::span<double>*> packaged_data_buffer,
    const BoundaryCorrection& boundary_correction,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_vars,
    const Variables<
        db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                         tmpl::size_t<Dim>, Frame::Inertial>>& volume_fluxes,
    const Variables<TemporaryTags>& volume_temporaries,
    const Variables<get_primitive_vars_tags_from_system<System>>* const
        volume_primitive_variables,
    const Element<Dim>& element, const Mesh<Dim>& volume_mesh,
    const std::unordered_map<
        std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim - 1>,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& mortar_meshes,
    const std::unordered_map<
        std::pair<Direction<Dim>, ElementId<Dim>>,
        std::array<Spectral::MortarSize, Dim - 1>,
        boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>& mortar_sizes,
    const TimeStepId& temporal_id,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
        moving_mesh_map,
    const std::optional<tnsr::I<DataVector, Dim>>& volume_mesh_velocity,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& volume_inverse_jacobian,
    const PackageDataVolumeArgs&... package_data_volume_args) {
  using variables_tags = typename System::variables_tag::tags_list;
  using flux_variables = typename System::flux_variables;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, flux_variables,
                                       tmpl::size_t<Dim>, Frame::Inertial>;
  using temporary_tags_for_face =
      typename BoundaryCorrection::dg_package_data_temporary_tags;
  using primitive_tags_for_face = typename detail::get_primitive_vars<
      System::has_primitive_and_conservative_vars>::
      template f<BoundaryCorrection>;
  using mortar_tags_list = typename BoundaryCorrection::dg_package_field_tags;

  using dg_package_data_projected_tags =
      tmpl::append<variables_tags, fluxes_tags, temporary_tags_for_face,
                   primitive_tags_for_face>;
  using FieldsOnFace = Variables<tmpl::remove_duplicates<tmpl::push_back<
      tmpl::append<dg_package_data_projected_tags,
                   detail::inverse_spatial_metric_tag<System>>,
      detail::OneOverNormalVectorMagnitude, detail::NormalVector<Dim>>>>;
  FieldsOnFace fields_on_face{};
  std::optional<tnsr::I<DataVector, Dim>> face_mesh_velocity{};
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    const Mesh<Dim - 1> face_mesh =
        volume_mesh.slice_away(direction.dimension());

    // The face_temporaries buffer is guaranteed to be big enough because we
    // allocated it in ComputeTimeDerivative with the max number of grid points
    // over all faces. We still check anyways in Debug mode to be safe
    ASSERT(face_temporaries->size() >=
               FieldsOnFace::number_of_independent_components *
                   face_mesh.number_of_grid_points(),
           "The buffer for computing fields on faces which was allocated in "
           "ComputeTimeDerivative is not large enough. It's size is "
               << face_temporaries->size() << ", but needs to be at least "
               << FieldsOnFace::number_of_independent_components *
                      face_mesh.number_of_grid_points());

    fields_on_face.set_data_ref(face_temporaries->data(),
                                FieldsOnFace::number_of_independent_components *
                                    face_mesh.number_of_grid_points());

    // We may not need to bring the volume fluxes or temporaries to the
    // boundary since that depends on the specific boundary correction we
    // are using. Silence compilers warnings about them being unused.
    (void)volume_fluxes;
    (void)volume_temporaries;

    // This does the following:
    //
    // 1. Use a helper function to get data onto the faces. Done either by
    //    slicing (Gauss-Lobatto points) or interpolation (Gauss points).
    //    This is done using the `project_contiguous_data_to_boundary` and
    //    `project_tensors_to_boundary` functions.
    //
    // 2. Invoke the boundary correction to get the packaged data. Note
    //    that this is done on the *face* and NOT the mortar.
    //
    // 3. Project the packaged data onto the DG mortars (these might need
    //    re-projection onto subcell mortars later).

    // Perform step 1
    project_contiguous_data_to_boundary(make_not_null(&fields_on_face),
                                        volume_evolved_vars, volume_mesh,
                                        direction);
    if constexpr (tmpl::size<fluxes_tags>::value != 0) {
      project_contiguous_data_to_boundary(make_not_null(&fields_on_face),
                                          volume_fluxes, volume_mesh,
                                          direction);
    }
    if constexpr (tmpl::size<tmpl::append<
                      temporary_tags_for_face,
                      detail::inverse_spatial_metric_tag<System>>>::value !=
                  0) {
      project_tensors_to_boundary<tmpl::append<
          temporary_tags_for_face, detail::inverse_spatial_metric_tag<System>>>(
          make_not_null(&fields_on_face), volume_temporaries, volume_mesh,
          direction);
    }
    if constexpr (System::has_primitive_and_conservative_vars and
                  tmpl::size<primitive_tags_for_face>::value != 0) {
      ASSERT(volume_primitive_variables != nullptr,
             "The volume primitive variables are not set even though the "
             "system has primitive variables.");
      project_tensors_to_boundary<primitive_tags_for_face>(
          make_not_null(&fields_on_face), *volume_primitive_variables,
          volume_mesh, direction);
    } else {
      (void)volume_primitive_variables;
    }
    if (volume_mesh_velocity.has_value()) {
      if (not face_mesh_velocity.has_value() or
          (*face_mesh_velocity)[0].size() !=
              face_mesh.number_of_grid_points()) {
        face_mesh_velocity =
            tnsr::I<DataVector, Dim>{face_mesh.number_of_grid_points()};
      }
      project_tensor_to_boundary(make_not_null(&*face_mesh_velocity),
                                 *volume_mesh_velocity, volume_mesh, direction);
    }

    // Normalize the normal vectors. We cache the unit normal covector For
    // flat geometry and static meshes.
    const bool mesh_is_moving = not moving_mesh_map.is_identity();
    if (auto& normal_covector_quantity =
            normal_covector_and_magnitude_ptr->at(direction);
        detail::has_inverse_spatial_metric_tag_v<System> or mesh_is_moving or
        not normal_covector_quantity.has_value()) {
      if (not normal_covector_quantity.has_value()) {
        normal_covector_quantity =
            Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<Dim>>>{
                fields_on_face.number_of_grid_points()};
      }
      tnsr::i<DataVector, Dim> volume_unnormalized_normal_covector{};

      for (size_t inertial_index = 0; inertial_index < Dim; ++inertial_index) {
        volume_unnormalized_normal_covector.get(inertial_index)
            .set_data_ref(const_cast<double*>(  // NOLINT
                              volume_inverse_jacobian
                                  .get(direction.dimension(), inertial_index)
                                  .data()),
                          volume_mesh.number_of_grid_points());
      }
      project_tensor_to_boundary(
          make_not_null(&get<evolution::dg::Tags::NormalCovector<Dim>>(
              *normal_covector_quantity)),
          volume_unnormalized_normal_covector, volume_mesh, direction);

      if (direction.side() == Side::Lower) {
        for (auto& normal_covector_component :
             get<evolution::dg::Tags::NormalCovector<Dim>>(
                 *normal_covector_quantity)) {
          normal_covector_component *= -1.0;
        }
      }

      detail::unit_normal_vector_and_covector_and_magnitude_impl<System>(
          make_not_null(&get<evolution::dg::Tags::MagnitudeOfNormal>(
              *normal_covector_quantity)),
          make_not_null(&get<evolution::dg::Tags::NormalCovector<Dim>>(
              *normal_covector_quantity)),
          make_not_null(&fields_on_face),
          get<evolution::dg::Tags::NormalCovector<Dim>>(
              *normal_covector_quantity));
    }

    // Perform step 2
    ASSERT(normal_covector_and_magnitude_ptr->at(direction).has_value(),
           "The magnitude of the normal vector and the unit normal "
           "covector have not been computed, even though they should "
           "have been. Direction: "
               << direction);

    const size_t total_face_size =
        face_mesh.number_of_grid_points() *
        Variables<mortar_tags_list>::number_of_independent_components;
    Variables<mortar_tags_list> packaged_data{};

    // This is the case where we only have one neighbor in this direction, so we
    // may or may not have to do any projection. If we don't have to do
    // projection, then we can use the local_mortar_data itself to calculate the
    // dg_package_data. However, if we need to project, then we hae to use the
    // packaged_data_buffer that was passed in.
    if (neighbors_in_direction.size() == 1) {
      const auto& neighbor = *neighbors_in_direction.begin();
      const auto mortar_id = std::pair{direction, neighbor};
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);

      // Have to use packaged_data_buffer
      if (Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
        // The face mesh will be assigned below along with ensuring the size of
        // the mortar data is correct
        packaged_data.set_data_ref(packaged_data_buffer->data(),
                                   total_face_size);
      } else {
        // Can use the local_mortar_data
        auto& local_mortar_data_opt =
            mortar_data_ptr->at(mortar_id).local_mortar_data();
        // If this isn't the first time, set the face mesh
        if (LIKELY(local_mortar_data_opt.has_value())) {
          local_mortar_data_opt->first = face_mesh;
        } else {
          // Otherwise we need to initialize the pair. If we don't do this, then
          // the DataVector will be non-owning which we don't want
          local_mortar_data_opt =
              std::optional{std::pair{face_mesh, DataVector{}}};
        }

        // Always set the time step ID
        mortar_data_ptr->at(mortar_id).time_step_id() = temporal_id;

        DataVector& local_mortar_data = local_mortar_data_opt->second;

        // Do a destructive resize to account for potential p-refinement
        local_mortar_data.destructive_resize(total_face_size);

        packaged_data.set_data_ref(local_mortar_data.data(),
                                   local_mortar_data.size());
      }
    } else {
      // In this case, we have multiple neighbors in this direction so all will
      // need to project their data which means we use the
      // packaged_data_buffer to calculate the dg_package_data
      packaged_data.set_data_ref(packaged_data_buffer->data(), total_face_size);
    }

    detail::dg_package_data<System>(
        make_not_null(&packaged_data), boundary_correction, fields_on_face,
        get<evolution::dg::Tags::NormalCovector<Dim>>(
            *normal_covector_and_magnitude_ptr->at(direction)),
        face_mesh_velocity, dg_package_data_projected_tags{},
        package_data_volume_args...);

    // Perform step 3
    // This will only do something if
    //  a) we have multiple neighbors in this direction
    // or
    //  b) the one (and only) neighbor in this direction needed projection
    for (const auto& neighbor : neighbors_in_direction) {
      const auto mortar_id = std::make_pair(direction, neighbor);
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);

      if (Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
        auto& local_mortar_data_opt =
            mortar_data_ptr->at(mortar_id).local_mortar_data();

        // If this isn't the first time, set the face mesh
        if (LIKELY(local_mortar_data_opt.has_value())) {
          local_mortar_data_opt->first = face_mesh;
        } else {
          // If we don't do this, then the DataVector will be non-owning which
          // we don't want
          local_mortar_data_opt =
              std::optional{std::pair{face_mesh, DataVector{}}};
        }

        // Set the time id since above we only set it for cases that didn't need
        // projection
        mortar_data_ptr->at(mortar_id).time_step_id() = temporal_id;

        DataVector& local_mortar_data = local_mortar_data_opt->second;

        // Do a destructive resize to account for potential p-refinement
        local_mortar_data.destructive_resize(
            mortar_mesh.number_of_grid_points() *
            Variables<mortar_tags_list>::number_of_independent_components);

        Variables<mortar_tags_list> projected_packaged_data{
            local_mortar_data.data(), local_mortar_data.size()};
        ::dg::project_to_mortar(make_not_null(&projected_packaged_data),
                                packaged_data, face_mesh, mortar_mesh,
                                mortar_size);
      }
    }
  }
}

template <typename System, size_t Dim, typename BoundaryCorrection,
          typename DbTagsList, typename... PackageDataVolumeTags>
void internal_mortar_data(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<gsl::span<double>*> face_temporaries,
    const gsl::not_null<gsl::span<double>*> packaged_data_buffer,
    const BoundaryCorrection& boundary_correction,
    const Variables<typename System::variables_tag::tags_list>
        evolved_variables,
    const Variables<
        db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                         tmpl::size_t<Dim>, Frame::Inertial>>& volume_fluxes,
    const Variables<
        typename System::compute_volume_time_derivative_terms::temporary_tags>&
        temporaries,
    const Variables<get_primitive_vars_tags_from_system<System>>* const
        primitive_vars,
    tmpl::list<PackageDataVolumeTags...> /*meta*/) {
  db::mutate<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
             evolution::dg::Tags::MortarData<Dim>>(
      [&boundary_correction, &face_temporaries, &packaged_data_buffer,
       &element = db::get<domain::Tags::Element<Dim>>(*box), &evolved_variables,
       &logical_to_inertial_inverse_jacobian =
           db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                 Frame::Inertial>>(*box),
       &mesh = db::get<domain::Tags::Mesh<Dim>>(*box),
       &mesh_velocity = db::get<domain::Tags::MeshVelocity<Dim>>(*box),
       &mortar_meshes = db::get<Tags::MortarMesh<Dim>>(*box),
       &mortar_sizes = db::get<Tags::MortarSize<Dim>>(*box),
       &moving_mesh_map = db::get<domain::CoordinateMaps::Tags::CoordinateMap<
           Dim, Frame::Grid, Frame::Inertial>>(*box),
       &primitive_vars, &temporaries,
       &time_step_id = db::get<::Tags::TimeStepId>(*box), &volume_fluxes](
          const auto normal_covector_and_magnitude_ptr,
          const auto mortar_data_ptr, const auto&... package_data_volume_args) {
        detail::internal_mortar_data_impl<System>(
            normal_covector_and_magnitude_ptr, mortar_data_ptr,
            face_temporaries, packaged_data_buffer, boundary_correction,
            evolved_variables, volume_fluxes, temporaries, primitive_vars,
            element, mesh, mortar_meshes, mortar_sizes, time_step_id,
            moving_mesh_map, mesh_velocity,
            logical_to_inertial_inverse_jacobian, package_data_volume_args...);
      },
      box, db::get<PackageDataVolumeTags>(*box)...);
}
}  // namespace evolution::dg::Actions::detail
