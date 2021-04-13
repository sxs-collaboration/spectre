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
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
///
/// \warning Make sure the two face meshes are oriented the same, i.e.
/// their dimensions align. This is facilitated by the `orientation`
/// passed to `domain::Initialization::create_initial_mesh`, for
/// example.
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
/// Project variables from a face to a mortar.
template <typename Tags, size_t Dim>
Variables<Tags> project_to_mortar(const Variables<Tags>& vars,
                                  const Mesh<Dim>& face_mesh,
                                  const Mesh<Dim>& mortar_mesh,
                                  const MortarSize<Dim>& mortar_size) noexcept {
  const auto projection_matrices = Spectral::projection_matrix_parent_to_child(
      face_mesh, mortar_mesh, mortar_size);
  return apply_matrices(projection_matrices, vars, face_mesh.extents());
}

/// \ingroup DiscontinuousGalerkinGroup
/// Project variables from a mortar to a face.
template <typename Tags, size_t Dim>
Variables<Tags> project_from_mortar(
    const Variables<Tags>& vars, const Mesh<Dim>& face_mesh,
    const Mesh<Dim>& mortar_mesh, const MortarSize<Dim>& mortar_size) noexcept {
  ASSERT(Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size),
         "project_from_mortar should not be called if the interface mesh and "
         "mortar mesh are identical. Please elide the copy instead.");
  const auto projection_matrices = Spectral::projection_matrix_child_to_parent(
      mortar_mesh, face_mesh, mortar_size);
  return apply_matrices(projection_matrices, vars, mortar_mesh.extents());
}

}  // namespace dg
