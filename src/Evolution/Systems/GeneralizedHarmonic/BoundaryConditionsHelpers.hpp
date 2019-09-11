// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace GeneralizedHarmonic {
namespace Actions {
namespace BoundaryConditions_detail {
template <size_t VolumeDim>
double min_characteristic_speed(
    const typename GeneralizedHarmonic::Tags::CharacteristicSpeeds<
        VolumeDim, Frame::Inertial>::type& char_speeds) noexcept {
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

// Eq. (2.20) of Kidder, Scheel & Teukolsky (KST) [gr-qc/0105031v1]
// Compute R_{ij} from D_(k,i,j) = (1/2) \partial_k g_(ij)
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> spatial_ricci_tensor_from_KST_vars(
    const tnsr::ijj<DataType, VolumeDim, Frame>& kst_var_D,
    const tnsr::ijkk<DataType, VolumeDim, Frame>& d_kst_var_D,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept {
  auto ricci = make_with_value<tnsr::ii<DataType, VolumeDim, Frame>>(
      inverse_spatial_metric, 0.);

  // New variable to avoid recomputing stuff in nested loops
  tnsr::ijj<DataType, VolumeDim, Frame> Dudd(
      get_size(get<0, 0>(inverse_spatial_metric)));

  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
        Dudd.get(k, i, j) = 0.;
        for (size_t l = 0; l < VolumeDim; ++l)
          Dudd.get(k, i, j) +=
              inverse_spatial_metric.get(k, l) * kst_var_D.get(l, i, j);
      }
    }
  }

  // New variable to avoid recomputing stuff in nested loops
  tnsr::ijk<DataType, VolumeDim, Frame> Dddu(
      get_size(get<0, 0>(inverse_spatial_metric)));
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        Dddu.get(i, j, k) = 0.;
        for (size_t l = 0; l < VolumeDim; ++l)
          Dddu.get(i, j, k) +=
              inverse_spatial_metric.get(k, l) * kst_var_D.get(i, j, l);
      }
    }
  }

  // New variable to avoid recomputing stuff in nested loops
  tnsr::i<DataType, VolumeDim, Frame> Dtr1(
      get_size(get<0, 0>(inverse_spatial_metric)));
  //  kst_var_D.get(i,m,n)*inverse_spatial_metric.get(m,n)
  for (size_t i = 0; i < VolumeDim; ++i) {
    Dtr1.get(i) = 0.;
    for (size_t j = 0; j < VolumeDim; ++j)
      Dtr1.get(i) += Dddu.get(i, j, j);
  }
  //   kst_var_D.get(m,n,i)*inverse_spatial_metric.get(m,n)
  tnsr::i<DataType, VolumeDim, Frame> Dtr2(
      get_size(get<0, 0>(inverse_spatial_metric)));
  for (size_t i = 0; i < VolumeDim; ++i) {
    Dtr2.get(i) = 0.;
    for (size_t j = 0; j < VolumeDim; ++j)
      Dtr2.get(i) += Dudd.get(j, j, i);
  }

  const auto Dtr1up = raise_or_lower_index(Dtr1, inverse_spatial_metric);
  const auto Dtr2up = raise_or_lower_index(Dtr2, inverse_spatial_metric);

  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
      ricci.get(i, j) = 0.;
      for (size_t p = 0; p < VolumeDim; ++p) {
        ricci.get(i, j) += (kst_var_D.get(p, i, j) - kst_var_D.get(i, j, p) -
                            kst_var_D.get(j, i, p)) *
                           (2.0 * Dtr2up.get(p) - Dtr1up.get(p));
        for (size_t q = 0; q < VolumeDim; ++q) {
          ricci.get(i, j) +=
              0.5 * inverse_spatial_metric.get(p, q) *
                  (d_kst_var_D.get(j, q, p, i) + d_kst_var_D.get(i, q, p, j) -
                   d_kst_var_D.get(j, i, p, q) - d_kst_var_D.get(i, j, p, q) +
                   d_kst_var_D.get(p, i, q, j) + d_kst_var_D.get(p, j, q, i) -
                   2.0 * d_kst_var_D.get(q, p, i, j)) +
              Dddu.get(i, p, q) * Dddu.get(j, q, p) +
              2.0 * Dudd.get(p, i, q) * Dddu.get(p, j, q) -
              2.0 * Dudd.get(p, q, i) * Dudd.get(q, p, j);
        }
      }
    }
  }

  return ricci;
}

// Eq. (2.19) of Kidder, Scheel & Teukolsky (KST) [gr-qc/0105031v1]
// Compute \Gamma^k_{ij} from D_(k,i,j) = (1/2) \partial_k g_(ij)
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::Ijj<DataType, VolumeDim, Frame>
spatial_christoffel_second_kind_from_KST_vars(
    const tnsr::ijj<DataType, VolumeDim, Frame>& kst_var_D,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept {
  tnsr::ijj<DataType, VolumeDim, Frame> christoffel_first_kind{
      get_size(get<0, 0>(inverse_spatial_metric))};
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
        christoffel_first_kind.get(k, i, j) = kst_var_D.get(i, j, k) +
                                              kst_var_D.get(j, i, k) -
                                              kst_var_D.get(k, i, j);
      }
    }
  }
  auto christoffel_second_kind =
      make_with_value<tnsr::Ijj<DataType, VolumeDim, Frame>>(
          inverse_spatial_metric, 0.);
  // raise first index
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        for (size_t l = 0; l < VolumeDim; ++l) {
          christoffel_second_kind.get(k, i, j) +=
              inverse_spatial_metric.get(k, l) *
              christoffel_first_kind.get(l, i, j);
        }
      }
    }
  }
  return christoffel_second_kind;
}

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
    projection_tensor->get(i, i) = 0.;
    for (size_t j = 0; j < VolumeDim; ++j) {
      projection_tensor->get(i, j) =
          -normal_vector.get(i + 1) * normal_one_form.get(j + 1);
    }
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
    const tnsr::a<DataType, VolumeDim, Frame>& /* unit_normal */,
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
            sign * unit_interface_normal_vector.get(c) *
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
