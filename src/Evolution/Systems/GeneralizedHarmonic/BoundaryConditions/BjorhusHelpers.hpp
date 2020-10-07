// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
namespace BoundaryConditions_detail {
template <size_t VolumeDim>
double min_characteristic_speed(
    const std::array<DataVector, 4>& char_speeds) noexcept {
  std::array<double, 4> min_speeds{
      {min(char_speeds.at(0)), min(char_speeds.at(1)), min(char_speeds.at(2)),
       min(char_speeds.at(3))}};
  return *std::min_element(min_speeds.begin(), min_speeds.end());
}

template <typename T, typename DataType>
T set_bc_when_char_speed_is_negative(const T& rhs_char_dt_u,
                                     const T& desired_bc_dt_u,
                                     const DataType& char_speed_u) noexcept {
  auto bc_dt_u = rhs_char_dt_u;
  auto it1 = bc_dt_u.begin();
  auto it2 = desired_bc_dt_u.begin();
  for (; it2 != desired_bc_dt_u.end(); ++it1, ++it2) {
    for (size_t i = 0; i < it1->size(); ++i) {
      if (char_speed_u[i] < 0.) {
        (*it1)[i] = (*it2)[i];
      }
    }
  }
  return bc_dt_u;
}
}  // namespace BoundaryConditions_detail
}  // namespace Actions

// Spatial projection tensors
template <size_t VolumeDim, typename Frame, typename DataType, typename RetType>
void spatial_projection_tensor(
    const gsl::not_null<RetType*> projection_tensor,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, VolumeDim, Frame>& normal_vector) noexcept {
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {
      projection_tensor->get(i, j) =
          inverse_spatial_metric.get(i, j) -
          normal_vector.get(i + 1) * normal_vector.get(j + 1);
    }
  }
}
template <size_t VolumeDim, typename Frame, typename DataType, typename RetType>
void spatial_projection_tensor(
    const gsl::not_null<RetType*> projection_tensor,
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::a<DataType, VolumeDim, Frame>& normal_one_form) noexcept {
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {
      projection_tensor->get(i, j) =
          spatial_metric.get(i, j) -
          normal_one_form.get(i + 1) * normal_one_form.get(j + 1);
    }
  }
}
template <size_t VolumeDim, typename Frame, typename DataType, typename RetType>
void spatial_projection_tensor(
    const gsl::not_null<RetType*> projection_tensor,
    const tnsr::A<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame>& normal_one_form) noexcept {
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = 0; j < VolumeDim; ++j) {
      projection_tensor->get(i, j) =
          -normal_vector.get(i + 1) * normal_one_form.get(j + 1);
    }
    projection_tensor->get(i, i) += 1.;
  }
}

// This computes U8(+/-)_{ij} = [P^(a_i P^b)_j - (1/2) P_{ij} P^{ab}] *
//                              [E_{ab} -/+ \epsilon_{a}{}^{cd} n_d B_{cb}]
//
// Note that U8+_{ij} is proportional to the Newman-Penrose Psi4
//       and U8-_{ij} is proportional to the Newman-Penrose Psi0
//
// Here
// U8(a,b)     = U8_{ab}
// ni(a)       = n_a      (unit normal)
// nI(a)       = n^a
// PIJ(a,b)    = P^{ab}   (projection operator = g^{ab} - n^a n^b)
// Pij(a,b)    = P_{ab}   (projection operator = g_{ab} - n_a n_b)
// PIj(a,b)    = P^a_b    (projection operator = g^a_b  - n^a n_b)
// K(a,b)      = K_{ab}
// Ricci(a,b)  = Ricci_{ab}
// Invg(a,b)   = g^{ab}
// CdK(a,b)(c) = \nabla_c K_{ab}
template <size_t VolumeDim, typename Frame, typename DataType, typename RetType>
void weyl_propagating(
    const gsl::not_null<RetType*> U8,
    const tnsr::ii<DataType, VolumeDim, Frame>& ricci,
    const tnsr::ii<DataType, VolumeDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, VolumeDim, Frame>& CdK,
    const tnsr::A<DataType, VolumeDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, VolumeDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, VolumeDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, VolumeDim, Frame>& projection_Ij,
    const int sign) noexcept {
  // Fill temp with unprojected quantity
  tnsr::ii<DataType, VolumeDim, Frame> temp(
      get_size(get<0>(unit_interface_normal_vector)));
  for (size_t a = 0; a < VolumeDim; ++a) {
    for (size_t b = a; b < VolumeDim; ++b) {  // Symmetry
      temp.get(a, b) = ricci.get(a, b);
      for (size_t c = 0; c < VolumeDim; ++c) {
        temp.get(a, b) -=
            sign * unit_interface_normal_vector.get(1 + c) *
            (CdK.get(c, a, b) - 0.5 * (CdK.get(b, a, c) + CdK.get(a, b, c)));

        for (size_t d = 0; d < VolumeDim; ++d) {
          temp.get(a, b) +=
              inverse_spatial_metric.get(c, d) *
              (extrinsic_curvature.get(a, b) * extrinsic_curvature.get(c, d) -
               extrinsic_curvature.get(a, c) * extrinsic_curvature.get(d, b));
        }
      }
    }
  }

  // Now project
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
      U8->get(i, j) = 0.;
      for (size_t a = 0; a < VolumeDim; ++a) {
        for (size_t b = 0; b < VolumeDim; ++b) {
          U8->get(i, j) +=
              (projection_Ij.get(a, i) * projection_Ij.get(b, j) -
               0.5 * projection_IJ.get(a, b) * projection_ij.get(i, j)) *
              temp.get(a, b);
        }
      }
    }
  }
}
}  // namespace GeneralizedHarmonic
