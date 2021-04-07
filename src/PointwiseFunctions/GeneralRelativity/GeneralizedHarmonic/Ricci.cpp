// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> ricci,
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept {
  destructive_resize_components(ricci, get_size(get<0, 0, 0>(phi)));

  TempBuffer<tmpl::list<::Tags::TempIjj<0, VolumeDim, Frame, DataType>,
                        ::Tags::TempijK<0, VolumeDim, Frame, DataType>,
                        ::Tags::Tempi<0, VolumeDim, Frame, DataType>,
                        ::Tags::Tempi<1, VolumeDim, Frame, DataType>,
                        ::Tags::TempI<0, VolumeDim, Frame, DataType>,
                        ::Tags::TempI<1, VolumeDim, Frame, DataType>>>
      local_buffer(get_size(get<0, 0>(inverse_spatial_metric)));

  // Note that this is (1/2) times \Phi^i_{jk} (without the trace)
  auto& half_spatial_phi_Ijj =
      get<::Tags::TempIjj<0, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
        half_spatial_phi_Ijj.get(k, i, j) =
            0.5 * inverse_spatial_metric.get(k, 0) * phi.get(0, 1 + i, 1 + j);
        for (size_t l = 1; l < VolumeDim; ++l) {
          half_spatial_phi_Ijj.get(k, i, j) +=
              0.5 * inverse_spatial_metric.get(k, l) * phi.get(l, 1 + i, 1 + j);
        }
      }
    }
  }

  // Note that this is (1/2) times \Phi_{ij}^k (without the trace)
  auto& half_spatial_phi_ijK =
      get<::Tags::TempijK<0, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        half_spatial_phi_ijK.get(i, j, k) =
            0.5 * inverse_spatial_metric.get(k, 0) * phi.get(i, 1 + j, 1);
        for (size_t l = 1; l < VolumeDim; ++l) {
          half_spatial_phi_ijK.get(i, j, k) +=
              0.5 * inverse_spatial_metric.get(k, l) * phi.get(i, 1 + j, 1 + l);
        }
      }
    }
  }

  // Note that these traces are actually (1/2) times d_k and b_k respectively
  auto& half_spatial_phi_trace_second_third_indices =
      get<::Tags::Tempi<0, VolumeDim, Frame, DataType>>(local_buffer);
  auto& half_spatial_phi_trace_first_second_indices =
      get<::Tags::Tempi<1, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t i = 0; i < VolumeDim; ++i) {
    half_spatial_phi_trace_second_third_indices.get(i) =
        half_spatial_phi_ijK.get(i, 0, 0);
    half_spatial_phi_trace_first_second_indices.get(i) =
        half_spatial_phi_Ijj.get(0, 0, i);
    for (size_t j = 1; j < VolumeDim; ++j) {
      half_spatial_phi_trace_second_third_indices.get(i) +=
          half_spatial_phi_ijK.get(i, j, j);
      half_spatial_phi_trace_first_second_indices.get(i) +=
          half_spatial_phi_Ijj.get(j, j, i);
    }
  }

  // Again, these traces are actually (1/2) times d^k and b^k respectively
  auto& half_spatial_phi_trace_second_third_indices_I =
      get<::Tags::TempI<0, VolumeDim, Frame, DataType>>(local_buffer);
  auto& half_spatial_phi_trace_first_second_indices_I =
      get<::Tags::TempI<1, VolumeDim, Frame, DataType>>(local_buffer);

  raise_or_lower_index(
      make_not_null(&half_spatial_phi_trace_second_third_indices_I),
      half_spatial_phi_trace_second_third_indices, inverse_spatial_metric);
  raise_or_lower_index(
      make_not_null(&half_spatial_phi_trace_first_second_indices_I),
      half_spatial_phi_trace_first_second_indices, inverse_spatial_metric);

  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
      ricci->get(i, j) = 0.;
      for (size_t k = 0; k < VolumeDim; ++k) {
        ricci->get(i, j) +=
            0.5 *
            (phi.get(i, 1 + j, 1 + k) + phi.get(j, 1 + i, 1 + k) -
             phi.get(k, 1 + i, 1 + j)) *
            (half_spatial_phi_trace_second_third_indices_I.get(k) -
             2.0 * half_spatial_phi_trace_first_second_indices_I.get(k));
        for (size_t l = 0; l < VolumeDim; ++l) {
          ricci->get(i, j) += 0.25 * inverse_spatial_metric.get(k, l) *
                                  (deriv_phi.get(j, l, 1 + k, 1 + i) +
                                   deriv_phi.get(i, l, 1 + k, 1 + j) -
                                   deriv_phi.get(j, i, 1 + k, 1 + l) -
                                   deriv_phi.get(i, j, 1 + k, 1 + l) +
                                   deriv_phi.get(k, i, 1 + l, 1 + j) +
                                   deriv_phi.get(k, j, 1 + l, 1 + i) -
                                   2.0 * deriv_phi.get(l, k, 1 + i, 1 + j)) +
                              half_spatial_phi_ijK.get(i, k, l) *
                                  half_spatial_phi_ijK.get(j, l, k) +
                              2.0 * half_spatial_phi_Ijj.get(k, i, l) *
                                  half_spatial_phi_ijK.get(k, j, l) -
                              2.0 * half_spatial_phi_Ijj.get(k, l, i) *
                                  half_spatial_phi_Ijj.get(l, k, j);
        }
      }
    }
  }
}

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> spatial_ricci_tensor(
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept {
  tnsr::ii<DataType, VolumeDim, Frame> ricci{};
  GeneralizedHarmonic::spatial_ricci_tensor<VolumeDim, Frame, DataType>(
      make_not_null(&ricci), phi, deriv_phi, inverse_spatial_metric);
  return ricci;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void GeneralizedHarmonic::spatial_ricci_tensor(                \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          ricci,                                                          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& deriv_phi,   \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_spatial_metric) noexcept;                               \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::spatial_ricci_tensor(                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& deriv_phi,   \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
