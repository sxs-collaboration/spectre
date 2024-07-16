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

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
  p | face_mesh;
}

template <size_t Dim>
bool operator==(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return lhs.mortar_data == rhs.mortar_data and
         lhs.face_normal_magnitude == rhs.face_normal_magnitude and
         lhs.face_det_jacobian == rhs.face_det_jacobian and
         lhs.volume_det_inv_jacobian == rhs.volume_det_inv_jacobian and
         lhs.face_mesh == rhs.face_mesh;
}

template <size_t Dim>
bool operator!=(const MortarData<Dim>& lhs, const MortarData<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const MortarData<Dim>& mortar_data) {
  os << "Mortar data: " << mortar_data.mortar_data << "\n";
  os << "Face normal magnitude: " << mortar_data.face_normal_magnitude << "\n";
  os << "Face det(J): " << mortar_data.face_det_jacobian << "\n";
  os << "Face mesh: " << mortar_data.face_mesh << "\n";
  os << "Volume det(invJ): " << mortar_data.volume_det_inv_jacobian << "\n";
  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template class MortarData<DIM(data)>;                       \
  template bool operator==(const MortarData<DIM(data)>& lhs,  \
                           const MortarData<DIM(data)>& rhs); \
  template bool operator!=(const MortarData<DIM(data)>& lhs,  \
                           const MortarData<DIM(data)>& rhs); \
  template std::ostream& operator<<(std::ostream& os,         \
                                    const MortarData<DIM(data)>& mortar_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
