// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions::detail {
template <typename Metavariables, size_t Dim, typename BoundaryCorrection,
          typename TemporaryTags, typename DbTagsList, typename VariablesTags,
          typename... PackageDataVolumeTags>
void internal_mortar_data(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const BoundaryCorrection& boundary_correction,
    const Variables<VariablesTags>& volume_evolved_vars,
    const Variables<db::wrap_tags_in<
        ::Tags::Flux, typename Metavariables::system::flux_variables,
        tmpl::size_t<Dim>, Frame::Inertial>>& volume_fluxes,
    const Variables<TemporaryTags>& volume_temporaries) noexcept {
  using system = typename Metavariables::system;
  using variables_tags = typename system::variables_tag::tags_list;
  using flux_variables = typename system::flux_variables;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, flux_variables,
                                       tmpl::size_t<Metavariables::volume_dim>,
                                       Frame::Inertial>;
  using temporary_tags_for_face =
      typename BoundaryCorrection::dg_package_data_temporary_tags;
  using primitive_tags_for_face = typename detail::get_primitive_vars<
      system::has_primitive_and_conservative_vars>::
      template f<BoundaryCorrection>;
  using mortar_tags_list = typename BoundaryCorrection::dg_package_field_tags;
  static_assert(
      std::is_same_v<VariablesTags, variables_tags>,
      "The evolved variables passed in should match the evolved variables of "
      "the system.");

  const Element<Dim>& element = db::get<domain::Tags::Element<Dim>>(*box);
  const Mesh<Dim>& volume_mesh = db::get<domain::Tags::Mesh<Dim>>(*box);
  const auto& face_meshes =
      get<domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                  domain::Tags::Mesh<Dim - 1>>>(*box);
  const auto& mortar_meshes = db::get<Tags::MortarMesh<Dim>>(*box);
  const auto& mortar_sizes = db::get<Tags::MortarSize<Dim>>(*box);
  const TimeStepId& temporal_id = db::get<::Tags::TimeStepId>(*box);
  const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
      moving_mesh_map =
          db::get<domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                              Frame::Inertial>>(
              *box);
  const std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>&
      unnormalized_normal_covectors = db::get<domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>(*box);

  const std::unordered_map<Direction<Dim>,
                           std::optional<tnsr::I<DataVector, Dim>>>&
      face_mesh_velocities = db::get<domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          domain::Tags::MeshVelocity<Dim, Frame::Inertial>>>(*box);

  using dg_package_data_projected_tags =
      tmpl::append<variables_tags, fluxes_tags, temporary_tags_for_face,
                   primitive_tags_for_face>;
  Variables<tmpl::remove_duplicates<tmpl::push_back<
      tmpl::append<dg_package_data_projected_tags,
                   detail::inverse_spatial_metric_tag<system>>,
      detail::OneOverNormalVectorMagnitude, detail::NormalVector<Dim>>>>
      fields_on_face{};
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

    const auto internal_mortars = [&box, &boundary_correction, &element,
                                   &face_mesh_velocities, &fields_on_face,
                                   &mortar_meshes, &mortar_sizes,
                                   &moving_mesh_map, &temporal_id,
                                   &unnormalized_normal_covectors,
                                   &volume_evolved_vars, &volume_fluxes,
                                   &volume_temporaries,
                                   &volume_mesh](const Mesh<Dim - 1>& face_mesh,
                                                 const Direction<Dim>&
                                                     local_direction) noexcept {
      // We may not need to bring the volume fluxes or temporaries to the
      // boundary since that depends on the specific boundary correction we
      // are using. Silence compilers warnings about them being unused.
      (void)volume_fluxes;
      (void)volume_temporaries;
      // We only need the box for volume tags and primitive tags. Silence
      // unused variable compiler warnings in the case where we don't have
      // both of those.
      (void)box;

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
                        detail::inverse_spatial_metric_tag<system>>>::value !=
                    0) {
        project_tensors_to_boundary<
            tmpl::append<temporary_tags_for_face,
                         detail::inverse_spatial_metric_tag<system>>>(
            make_not_null(&fields_on_face), volume_temporaries, volume_mesh,
            local_direction);
      }
      if constexpr (system::has_primitive_and_conservative_vars and
                    tmpl::size<primitive_tags_for_face>::value != 0) {
        project_tensors_to_boundary<primitive_tags_for_face>(
            make_not_null(&fields_on_face),
            db::get<typename system::primitive_variables_tag>(*box),
            volume_mesh, local_direction);
      }

      // Normalize the normal vectors. In the case of a time-independent
      // mesh with no inverse spatial metric tag (i.e. flat background) we
      // could cache the normal vectors as simple tags in the DataBox as an
      // optimization. This will require making the
      // Normalized<UnnormalizedFaceNormal> a simple tag instead of a
      // compute tag.
      db::mutate<
          evolution::dg::Tags::InternalFace::NormalCovectorAndMagnitude<Dim>>(
          box, [&fields_on_face, &local_direction, &moving_mesh_map,
                &unnormalized_normal_covectors](
                   const auto normal_covector_and_magnitude_ptr) noexcept {
            detail::unit_normal_vector_and_covector_and_magnitude<system>(
                normal_covector_and_magnitude_ptr,
                make_not_null(&fields_on_face), local_direction,
                unnormalized_normal_covectors, moving_mesh_map);
          });

      // Perform step 2
      ASSERT(
          db::get<evolution::dg::Tags::InternalFace::NormalCovectorAndMagnitude<
              Dim>>(*box)
              .at(local_direction)
              .has_value(),
          "The magnitude of the normal vector and the unit normal "
          "covector have not been computed, even though they should "
          "have been. Direction: "
              << local_direction);

      Variables<mortar_tags_list> packaged_data{
          face_mesh.number_of_grid_points()};
      // The DataBox is passed in for retrieving the `volume_tags`
      const double max_abs_char_speed_on_face = detail::dg_package_data<system>(
          make_not_null(&packaged_data), boundary_correction, fields_on_face,
          get<evolution::dg::Tags::InternalFace::NormalCovector<Dim>>(
              *db::get<evolution::dg::Tags::InternalFace::
                           NormalCovectorAndMagnitude<Dim>>(*box)
                   .at(local_direction)),
          face_mesh_velocities.at(local_direction), *box,
          typename BoundaryCorrection::dg_package_data_volume_tags{},
          dg_package_data_projected_tags{});
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
            ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
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
        db::mutate<Tags::MortarData<Dim>>(
            box, [&face_mesh, &mortar_id, &temporal_id,
                  &type_erased_boundary_data_on_mortar](
                     const auto mortar_data_ptr) noexcept {
              mortar_data_ptr->at(mortar_id).insert_local_mortar_data(
                  temporal_id, face_mesh,
                  std::move(type_erased_boundary_data_on_mortar));
            });
      }
    };

    if (fields_on_face.number_of_grid_points() !=
        face_meshes.at(direction).number_of_grid_points()) {
      fields_on_face.initialize(
          face_meshes.at(direction).number_of_grid_points());
    }
    internal_mortars(face_meshes.at(direction), direction);

    if (element.neighbors().count(direction.opposite()) != 0) {
      if (fields_on_face.number_of_grid_points() !=
          face_meshes.at(direction.opposite()).number_of_grid_points()) {
        fields_on_face.initialize(
            face_meshes.at(direction.opposite()).number_of_grid_points());
      }
      internal_mortars(face_meshes.at(direction.opposite()),
                       direction.opposite());
    }
  }
}
}  // namespace evolution::dg::Actions::detail
