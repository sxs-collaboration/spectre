// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

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
    const PackageDataVolumeArgs&... package_data_volume_args) noexcept {
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
  Variables<tmpl::remove_duplicates<tmpl::push_back<
      tmpl::append<dg_package_data_projected_tags,
                   detail::inverse_spatial_metric_tag<System>>,
      detail::OneOverNormalVectorMagnitude, detail::NormalVector<Dim>>>>
      fields_on_face{};
  std::optional<tnsr::I<DataVector, Dim>> face_mesh_velocity{};
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    (void)neighbors_in_direction;  // unused variable
    // In order to reduce memory allocations we handle both the upper and
    // lower neighbors in each direction together since computing the
    // contributions to the faces is guaranteed to require the same number of
    // grid points.
    if (direction.side() == Side::Upper and
        element.neighbors().count(direction.opposite()) != 0) {
      continue;
    }

    const auto internal_mortars = [&](const Mesh<Dim - 1>& face_mesh,
                                      const Direction<Dim>&
                                          local_direction) noexcept {
      // We may not need to bring the volume fluxes or temporaries to the
      // boundary since that depends on the specific boundary correction we
      // are using. Silence compilers warnings about them being unused.
      (void)volume_fluxes;
      (void)volume_temporaries;

      // This helper does the following:
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
                                          local_direction);
      if constexpr (tmpl::size<fluxes_tags>::value != 0) {
        project_contiguous_data_to_boundary(make_not_null(&fields_on_face),
                                            volume_fluxes, volume_mesh,
                                            local_direction);
      }
      if constexpr (tmpl::size<tmpl::append<
                        temporary_tags_for_face,
                        detail::inverse_spatial_metric_tag<System>>>::value !=
                    0) {
        project_tensors_to_boundary<
            tmpl::append<temporary_tags_for_face,
                         detail::inverse_spatial_metric_tag<System>>>(
            make_not_null(&fields_on_face), volume_temporaries, volume_mesh,
            local_direction);
      }
      if constexpr (System::has_primitive_and_conservative_vars and
                    tmpl::size<primitive_tags_for_face>::value != 0) {
        ASSERT(volume_primitive_variables != nullptr,
               "The volume primitive variables are not set even though the "
               "system has primitive variables.");
        project_tensors_to_boundary<primitive_tags_for_face>(
            make_not_null(&fields_on_face), *volume_primitive_variables,
            volume_mesh, local_direction);
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
                                   *volume_mesh_velocity, volume_mesh,
                                   local_direction);
      }

      // Normalize the normal vectors. We cache the unit normal covector For
      // flat geometry and static meshes.
      const bool mesh_is_moving = not moving_mesh_map.is_identity();
      if (auto& normal_covector_quantity =
              normal_covector_and_magnitude_ptr->at(local_direction);
          detail::has_inverse_spatial_metric_tag_v<System> or mesh_is_moving or
          not normal_covector_quantity.has_value()) {
        if (not normal_covector_quantity.has_value()) {
          normal_covector_quantity =
              Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>{
                  fields_on_face.number_of_grid_points()};
        }
        tnsr::i<DataVector, Dim> volume_unnormalized_normal_covector{};

        for (size_t inertial_index = 0; inertial_index < Dim;
             ++inertial_index) {
          volume_unnormalized_normal_covector.get(inertial_index)
              .set_data_ref(
                  const_cast<double*>(  // NOLINT
                      volume_inverse_jacobian
                          .get(local_direction.dimension(), inertial_index)
                          .data()),
                  volume_mesh.number_of_grid_points());
        }
        project_tensor_to_boundary(
            make_not_null(&get<evolution::dg::Tags::NormalCovector<Dim>>(
                *normal_covector_quantity)),
            volume_unnormalized_normal_covector, volume_mesh, local_direction);

        if (local_direction.side() == Side::Lower) {
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
      ASSERT(normal_covector_and_magnitude_ptr->at(local_direction).has_value(),
             "The magnitude of the normal vector and the unit normal "
             "covector have not been computed, even though they should "
             "have been. Direction: "
                 << local_direction);

      Variables<mortar_tags_list> packaged_data{
          face_mesh.number_of_grid_points()};
      // The DataBox is passed in for retrieving the `volume_tags`
      const double max_abs_char_speed_on_face = detail::dg_package_data<System>(
          make_not_null(&packaged_data), boundary_correction, fields_on_face,
          get<evolution::dg::Tags::NormalCovector<Dim>>(
              *normal_covector_and_magnitude_ptr->at(local_direction)),
          face_mesh_velocity, dg_package_data_projected_tags{},
          package_data_volume_args...);
      (void)max_abs_char_speed_on_face;

      // Perform step 3
      const auto& neighbors_in_local_direction =
          element.neighbors().at(local_direction);
      for (const auto& neighbor : neighbors_in_local_direction) {
        const auto mortar_id = std::make_pair(local_direction, neighbor);
        const auto& mortar_mesh = mortar_meshes.at(mortar_id);
        const auto& mortar_size = mortar_sizes.at(mortar_id);

        // Project the data from the face to the mortar.
        // Where no projection is necessary we `std::move` the data
        // directly to avoid a copy. We can't move the data or modify it
        // in-place when projecting, because in that case the face may
        // touch two mortars so we need to keep the data around.
        auto boundary_data_on_mortar =
            Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ? ::dg::project_to_mortar(packaged_data, face_mesh, mortar_mesh,
                                          mortar_size)
                : std::move(packaged_data);

        // Store the boundary data on this side of the mortar in a way
        // that is agnostic to the type of boundary correction used. This
        // currently requires an additional allocation that could be
        // eliminated either by:
        //
        // 1. Having non-owning Variables
        //
        // 2. Allow stealing the allocation out of a Variables (and
        //    inserting an allocation).
        std::vector<double> type_erased_boundary_data_on_mortar{
            boundary_data_on_mortar.data(),
            boundary_data_on_mortar.data() + boundary_data_on_mortar.size()};
        mortar_data_ptr->at(mortar_id).insert_local_mortar_data(
            temporal_id, face_mesh,
            std::move(type_erased_boundary_data_on_mortar));
      }
    };

    const Mesh<Dim - 1> face_mesh =
        volume_mesh.slice_away(direction.dimension());

    if (fields_on_face.number_of_grid_points() !=
        face_mesh.number_of_grid_points()) {
      fields_on_face.initialize(face_mesh.number_of_grid_points());
    }
    internal_mortars(face_mesh, direction);

    if (element.neighbors().count(direction.opposite()) != 0) {
      if (fields_on_face.number_of_grid_points() !=
          face_mesh.number_of_grid_points()) {
        fields_on_face.initialize(face_mesh.number_of_grid_points());
      }
      internal_mortars(face_mesh, direction.opposite());
    }
  }
}

template <typename System, size_t Dim, typename BoundaryCorrection,
          typename DbTagsList, typename... PackageDataVolumeTags>
void internal_mortar_data(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
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
      box,
      [&boundary_correction,
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
       &time_step_id = db::get<::Tags::TimeStepId>(*box),
       &volume_fluxes](const auto normal_covector_and_magnitude_ptr,
                       const auto mortar_data_ptr,
                       const auto&... package_data_volume_args) noexcept {
        detail::internal_mortar_data_impl<System>(
            normal_covector_and_magnitude_ptr, mortar_data_ptr,
            boundary_correction, evolved_variables, volume_fluxes, temporaries,
            primitive_vars, element, mesh, mortar_meshes, mortar_sizes,
            time_step_id, moving_mesh_map, mesh_velocity,
            logical_to_inertial_inverse_jacobian, package_data_volume_args...);
      },
      db::get<PackageDataVolumeTags>(*box)...);
}
}  // namespace evolution::dg::Actions::detail
