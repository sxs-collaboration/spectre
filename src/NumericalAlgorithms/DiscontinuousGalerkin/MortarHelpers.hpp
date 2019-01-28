// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>  // IWYU pragma: keep // for std::forward

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
/// \cond
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class OrientationMap;
// IWYU pragma: no_forward_declare Variables
/// \endcond

namespace dg {
/// \ingroup DiscontinuousGalerkinGroup
/// Find a mesh for a mortar capable of representing data from either
/// of two faces.
template <size_t Dim>
Mesh<Dim> mortar_mesh(const Mesh<Dim>& face_mesh1,
                      const Mesh<Dim>& face_mesh2) noexcept;

/// \ingroup DiscontinuousGalerkinGroup
/// Determine the size of the mortar (i.e., the part of the face it
/// covers) for communicating with a neighbor.  This is the size
/// relative to the size of \p self, and will not generally agree with
/// that determined by \p neighbor.
template <size_t Dim>
std::array<Spectral::MortarSize, Dim - 1> mortar_size(
    const ElementId<Dim>& self, const ElementId<Dim>& neighbor,
    size_t dimension, const OrientationMap<Dim>& orientation) noexcept;

/// \ingroup DiscontinuousGalerkinGroup
/// Project variables from a face to a mortar.
template <typename Tags, size_t Dim>
Variables<Tags> project_to_mortar(
    const Variables<Tags>& vars,
    const Mesh<Dim>& face_mesh,
    const Mesh<Dim>& mortar_mesh,
    const std::array<Spectral::MortarSize, Dim>& mortar_size) noexcept {
  const Matrix identity{};
  auto projection_matrices = make_array<Dim>(std::cref(identity));

  const auto face_slice_meshes = face_mesh.slices();
  const auto mortar_slice_meshes = mortar_mesh.slices();
  for (size_t i = 0; i < Dim; ++i) {
    const auto& face_slice_mesh = gsl::at(face_slice_meshes, i);
    const auto& mortar_slice_mesh = gsl::at(mortar_slice_meshes, i);
    const auto& slice_size = gsl::at(mortar_size, i);
    if (slice_size != Spectral::MortarSize::Full or
        face_slice_mesh != mortar_slice_mesh) {
      gsl::at(projection_matrices, i) = projection_matrix_element_to_mortar(
          slice_size, mortar_slice_mesh, face_slice_mesh);
    }
  }
  return apply_matrices(projection_matrices, vars, face_mesh.extents());
}

/// \ingroup DiscontinuousGalerkinGroup
/// Project variables from a mortar to a face.
template <typename Tags, size_t Dim>
Variables<Tags> project_from_mortar(
    const Variables<Tags>& vars,
    const Mesh<Dim>& face_mesh,
    const Mesh<Dim>& mortar_mesh,
    const std::array<Spectral::MortarSize, Dim>& mortar_size) noexcept {
  ASSERT(face_mesh != mortar_mesh or
             alg::any_of(mortar_size,
                         [](const Spectral::MortarSize& size) noexcept {
                           return size != Spectral::MortarSize::Full;
                         }),
         "project_from_mortar should not be called if the interface mesh and "
         "mortar mesh are identical. Please elide the copy instead.");

  const Matrix identity{};
  auto projection_matrices = make_array<Dim>(std::cref(identity));

  const auto face_slice_meshes = face_mesh.slices();
  const auto mortar_slice_meshes = mortar_mesh.slices();
  for (size_t i = 0; i < Dim; ++i) {
    const auto& face_slice_mesh = gsl::at(face_slice_meshes, i);
    const auto& mortar_slice_mesh = gsl::at(mortar_slice_meshes, i);
    const auto& slice_size = gsl::at(mortar_size, i);
    if (slice_size != Spectral::MortarSize::Full or
        face_slice_mesh != mortar_slice_mesh) {
      gsl::at(projection_matrices, i) = projection_matrix_mortar_to_element(
          slice_size, face_slice_mesh, mortar_slice_mesh);
    }
  }
  return apply_matrices(projection_matrices, vars, mortar_mesh.extents());
}

namespace MortarHelpers_detail {
template <typename NormalDotNumericalFluxComputer,
          typename... NumericalFluxTags, typename... SelfTags,
          typename... PackagedTags>
void apply_normal_dot_numerical_flux(
    const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
        numerical_fluxes,
    const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
    const Variables<tmpl::list<SelfTags...>>& self_packaged_data,
    const Variables<tmpl::list<PackagedTags...>>&
        neighbor_packaged_data) noexcept {
  normal_dot_numerical_flux_computer(
      make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
      get<PackagedTags>(self_packaged_data)...,
      get<PackagedTags>(neighbor_packaged_data)...);
}
}  // namespace MortarHelpers_detail

/// \ingroup DiscontinuousGalerkinGroup
/// Compute the lifted data resulting from computing the numerical flux.
///
/// \details
/// This applies the numerical flux, projects the result to the face
/// mesh if necessary, and then lifts it to the volume (still
/// presented only on the face mesh as all other points are zero).
///
/// Projection must happen after the numerical flux calculation so
/// that the elements on either side of the mortar calculate the same
/// result.  Projection must happen before flux lifting because we
/// want the factor of the magnitude of the unit normal added during
/// the lift to cancel the Jacobian factor in integrals to preserve
/// conservation; this only happens if the two operations are done on
/// the same grid.
template <typename FluxCommTypes, typename NormalDotNumericalFluxComputer,
          size_t Dim, typename LocalData>
auto compute_boundary_flux_contribution(
    const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
    LocalData&& local_data,
    const typename FluxCommTypes::PackagedData& remote_data,
    const Mesh<Dim>& face_mesh, const Mesh<Dim>& mortar_mesh,
    const size_t extent_perpendicular_to_boundary,
    const std::array<Spectral::MortarSize, Dim>& mortar_size) noexcept
    -> db::item_type<
        db::remove_tag_prefix<typename FluxCommTypes::normal_dot_fluxes_tag>> {
  static_assert(cpp17::is_same_v<std::decay_t<LocalData>,
                                 typename FluxCommTypes::LocalData>,
                "Second argument must be a FluxCommTypes::LocalData");
  using variables_tag =
      db::remove_tag_prefix<typename FluxCommTypes::normal_dot_fluxes_tag>;
  db::item_type<db::add_tag_prefix<Tags::NormalDotNumericalFlux, variables_tag>>
      normal_dot_numerical_fluxes(mortar_mesh.number_of_grid_points(), 0.0);
  MortarHelpers_detail::apply_normal_dot_numerical_flux(
      make_not_null(&normal_dot_numerical_fluxes),
      normal_dot_numerical_flux_computer, local_data.mortar_data, remote_data);

  tmpl::for_each<db::get_variables_tags_list<variables_tag>>(
      [&normal_dot_numerical_fluxes, &local_data](const auto tag) noexcept {
        using Tag = tmpl::type_from<decltype(tag)>;
        auto& numerical_flux =
            get<Tags::NormalDotNumericalFlux<Tag>>(normal_dot_numerical_fluxes);
        const auto& local_flux =
            get<Tags::NormalDotFlux<Tag>>(local_data.mortar_data);
        for (size_t i = 0; i < numerical_flux.size(); ++i) {
          numerical_flux[i] -= local_flux[i];
        }
      });

  const bool refining =
      face_mesh != mortar_mesh or
      std::any_of(mortar_size.begin(), mortar_size.end(),
                  [](const Spectral::MortarSize s) noexcept {
                    return s != Spectral::MortarSize::Full;
                  });

  return dg::lift_flux(
      refining ? project_from_mortar(normal_dot_numerical_fluxes, face_mesh,
                                     mortar_mesh, mortar_size)
               : std::move(normal_dot_numerical_fluxes),
      extent_perpendicular_to_boundary,
      std::forward<LocalData>(local_data).magnitude_of_face_normal);
}
}  // namespace dg
