// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>  // IWYU pragma: keep // for std::forward

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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

template <size_t VolumeDim>
using MortarId = std::pair<::Direction<VolumeDim>, ElementId<VolumeDim>>;
template <size_t MortarDim>
using MortarSize = std::array<Spectral::MortarSize, MortarDim>;
template <size_t VolumeDim, typename ValueType>
using MortarMap = std::unordered_map<MortarId<VolumeDim>, ValueType,
                                     boost::hash<MortarId<VolumeDim>>>;

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
MortarSize<Dim - 1> mortar_size(
    const ElementId<Dim>& self, const ElementId<Dim>& neighbor,
    size_t dimension, const OrientationMap<Dim>& orientation) noexcept;

/// \ingroup DiscontinuousGalerkinGroup
/// Determine whether data on an element face needs to be projected to a mortar.
/// If no projection is necessary the data may be used on the mortar as-is.
template <size_t Dim>
bool needs_projection(const Mesh<Dim>& face_mesh, const Mesh<Dim>& mortar_mesh,
                      const MortarSize<Dim>& mortar_size) noexcept {
  return mortar_mesh != face_mesh or
      alg::any_of(mortar_size, [](const Spectral::MortarSize& size) noexcept {
        return size != Spectral::MortarSize::Full;
      });
}

/// \ingroup DiscontinuousGalerkinGroup
/// Project variables from a face to a mortar.
template <typename Tags, size_t Dim>
Variables<Tags> project_to_mortar(const Variables<Tags>& vars,
                                  const Mesh<Dim>& face_mesh,
                                  const Mesh<Dim>& mortar_mesh,
                                  const MortarSize<Dim>& mortar_size) noexcept {
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
    const Variables<Tags>& vars, const Mesh<Dim>& face_mesh,
    const Mesh<Dim>& mortar_mesh, const MortarSize<Dim>& mortar_size) noexcept {
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

}  // namespace dg
