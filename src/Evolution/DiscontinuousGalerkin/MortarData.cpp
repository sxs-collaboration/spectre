// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"

#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg {
template <size_t Dim>
void MortarData<Dim>::pup(PUP::er& p) {
  p | mortar_data;
  p | face_normal_magnitude;
  p | face_det_jacobian;
  p | volume_det_inv_jacobian;
  p | mortar_mesh;
  p | face_mesh;
  p | volume_mesh;
}

template <size_t Dim>
void p_project(
    const gsl::not_null<::evolution::dg::MortarData<Dim>*> mortar_data,
    const Mesh<Dim - 1>& new_mortar_mesh, const Mesh<Dim - 1>& new_face_mesh,
    const Mesh<Dim>& new_volume_mesh) {
  // nothing needs to be done in 1D as mortars/faces are a single point...
  if constexpr (Dim > 1) {
    if (mortar_data->mortar_data.has_value()) {
      const auto& old_mortar_mesh = mortar_data->mortar_mesh.value();
      if (old_mortar_mesh != new_mortar_mesh) {
        const auto mortar_projection_matrices =
            Spectral::p_projection_matrices(old_mortar_mesh, new_mortar_mesh);
        DataVector& vars = mortar_data->mortar_data.value();
        vars = apply_matrices(mortar_projection_matrices, vars,
                              old_mortar_mesh.extents());
        mortar_data->mortar_mesh = new_mortar_mesh;
      }
    }
    if (mortar_data->face_normal_magnitude.has_value()) {
      const auto& old_face_mesh = mortar_data->face_mesh.value();
      if (old_face_mesh != new_face_mesh) {
        const auto face_projection_matrices =
            Spectral::p_projection_matrices(old_face_mesh, new_face_mesh);
        DataVector& n = get(mortar_data->face_normal_magnitude.value());
        n = apply_matrices(face_projection_matrices, n,
                           old_face_mesh.extents());
        if (mortar_data->face_det_jacobian.has_value()) {
          DataVector& det_j = get(mortar_data->face_det_jacobian.value());
          det_j = apply_matrices(face_projection_matrices, det_j,
                                 old_face_mesh.extents());
        }
        mortar_data->face_mesh = new_face_mesh;
      }
    }
  }
  if (mortar_data->volume_det_inv_jacobian.has_value()) {
    const auto& old_volume_mesh = mortar_data->volume_mesh.value();
    if (old_volume_mesh != new_volume_mesh) {
      const auto volume_projection_matrices =
          Spectral::p_projection_matrices(old_volume_mesh, new_volume_mesh);
      DataVector& det_inv_j = get(mortar_data->volume_det_inv_jacobian.value());
      det_inv_j = apply_matrices(volume_projection_matrices, det_inv_j,
                                 old_volume_mesh.extents());
      mortar_data->volume_mesh = new_volume_mesh;
    }
  }
}

template <size_t Dim>
void p_project_only_mortar_data(
    const gsl::not_null<::evolution::dg::MortarData<Dim>*> mortar_data,
    const Mesh<Dim - 1>& new_mortar_mesh) {
  // nothing needs to be done in 1D as mortars are a single point...
  if constexpr (Dim > 1) {
    const auto& old_mortar_mesh = mortar_data->mortar_mesh.value();
    const auto mortar_projection_matrices =
        Spectral::p_projection_matrices(old_mortar_mesh, new_mortar_mesh);
    DataVector& vars = mortar_data->mortar_data.value();
    vars = apply_matrices(mortar_projection_matrices, vars,
                          old_mortar_mesh.extents());
    mortar_data->mortar_mesh = new_mortar_mesh;
  } else {
    (void)mortar_data;
    (void)new_mortar_mesh;
  }
}

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return lhs.mortar_data == rhs.mortar_data and
         lhs.face_normal_magnitude == rhs.face_normal_magnitude and
         lhs.face_det_jacobian == rhs.face_det_jacobian and
         lhs.volume_det_inv_jacobian == rhs.volume_det_inv_jacobian and
         lhs.mortar_mesh == rhs.mortar_mesh and
         lhs.face_mesh == rhs.face_mesh and lhs.volume_mesh == rhs.volume_mesh;
}

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const MortarData<Dim>& mortar_data) {
  os << "Mortar data: " << mortar_data.mortar_data << "\n";
  os << "Mortar mesh: " << mortar_data.mortar_mesh << "\n";
  os << "Face normal magnitude: " << mortar_data.face_normal_magnitude << "\n";
  os << "Face det(J): " << mortar_data.face_det_jacobian << "\n";
  os << "Face mesh: " << mortar_data.face_mesh << "\n";
  os << "Volume det(invJ): " << mortar_data.volume_det_inv_jacobian << "\n";
  os << "Volume mesh: " << mortar_data.volume_mesh << "\n";
  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                     \
  template class MortarData<DIM(data)>;                            \
  template void p_project(                                         \
      const gsl::not_null<::evolution::dg::MortarData<DIM(data)>*> \
          mortar_data,                                             \
      const Mesh<DIM(data) - 1>& new_mortar_mesh,                  \
      const Mesh<DIM(data) - 1>& new_face_mesh,                    \
      const Mesh<DIM(data)>& volume_mesh);                         \
  template void p_project_only_mortar_data(                        \
      const gsl::not_null<::evolution::dg::MortarData<DIM(data)>*> \
          mortar_data,                                             \
      const Mesh<DIM(data) - 1>& new_mortar_mesh);                 \
  template bool operator==(const MortarData<DIM(data)>& lhs,       \
                           const MortarData<DIM(data)>& rhs);      \
  template bool operator!=(const MortarData<DIM(data)>& lhs,       \
                           const MortarData<DIM(data)>& rhs);      \
  template std::ostream& operator<<(std::ostream& os,              \
                                    const MortarData<DIM(data)>& mortar_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
