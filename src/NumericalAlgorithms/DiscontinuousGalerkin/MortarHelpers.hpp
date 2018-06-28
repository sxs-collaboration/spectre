// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
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
/// Project variables from a face to a mortar.
template <typename Tags, size_t Dim>
Variables<Tags> project_to_mortar(const Variables<Tags>& vars,
                                  const Mesh<Dim>& face_mesh,
                                  const Mesh<Dim>& mortar_mesh) noexcept {
  const Matrix identity{};
  auto projection_matrices = make_array<Dim>(std::cref(identity));

  const auto face_slice_meshes = face_mesh.slices();
  const auto mortar_slice_meshes = mortar_mesh.slices();
  for (size_t i = 0; i < Dim; ++i) {
    const auto& face_slice_mesh = gsl::at(face_slice_meshes, i);
    const auto& mortar_slice_mesh = gsl::at(mortar_slice_meshes, i);
    if (face_slice_mesh != mortar_slice_mesh) {
      gsl::at(projection_matrices, i) = projection_matrix_element_to_mortar(
          Spectral::MortarSize::Full, mortar_slice_mesh, face_slice_mesh);
    }
  }
  return apply_matrices(projection_matrices, vars, face_mesh.extents());
}

/// \ingroup DiscontinuousGalerkinGroup
/// Project variables from a mortar to a face.
template <typename Tags, size_t Dim>
Variables<Tags> project_from_mortar(const Variables<Tags>& vars,
                                    const Mesh<Dim>& face_mesh,
                                    const Mesh<Dim>& mortar_mesh) noexcept {
  const Matrix identity{};
  auto projection_matrices = make_array<Dim>(std::cref(identity));

  const auto face_slice_meshes = face_mesh.slices();
  const auto mortar_slice_meshes = mortar_mesh.slices();
  for (size_t i = 0; i < Dim; ++i) {
    const auto& face_slice_mesh = gsl::at(face_slice_meshes, i);
    const auto& mortar_slice_mesh = gsl::at(mortar_slice_meshes, i);
    if (face_slice_mesh != mortar_slice_mesh) {
      gsl::at(projection_matrices, i) = projection_matrix_mortar_to_element(
          Spectral::MortarSize::Full, face_slice_mesh, mortar_slice_mesh);
    }
  }
  return apply_matrices(projection_matrices, vars, mortar_mesh.extents());
}
}  // namespace dg
