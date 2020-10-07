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

/// \cond
namespace GeneralizedHarmonic {
template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> ricci,
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*ricci)) != get_size(get<0, 0, 0>(phi)))) {
    *ricci = tnsr::ii<DataType, VolumeDim, Frame>(get_size(get<0, 0, 0>(phi)));
  }

  TempBuffer<tmpl::list<::Tags::TempIjj<0, VolumeDim, Frame, DataType>,
                        ::Tags::Tempijk<0, VolumeDim, Frame, DataType>,
                        ::Tags::Tempi<0, VolumeDim, Frame, DataType>,
                        ::Tags::Tempi<1, VolumeDim, Frame, DataType>,
                        ::Tags::TempI<0, VolumeDim, Frame, DataType>,
                        ::Tags::TempI<1, VolumeDim, Frame, DataType>>>
      local_buffer(get_size(get<0, 0>(inverse_spatial_metric)));

  // New variable to avoid recomputing in nested loops
  auto& phi_uplolo =
      get<::Tags::TempIjj<0, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
        phi_uplolo.get(k, i, j) = 0.;
        for (size_t l = 0; l < VolumeDim; ++l) {
          phi_uplolo.get(k, i, j) +=
              0.5 * inverse_spatial_metric.get(k, l) * phi.get(l, 1 + i, 1 + j);
        }
      }
    }
  }

  // New variable to avoid recomputing in nested loops
  auto& phi_loloup =
      get<::Tags::Tempijk<0, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t k = 0; k < VolumeDim; ++k) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        phi_loloup.get(i, j, k) = 0.;
        for (size_t l = 0; l < VolumeDim; ++l) {
          phi_loloup.get(i, j, k) +=
              0.5 * inverse_spatial_metric.get(k, l) * phi.get(i, 1 + j, 1 + l);
        }
      }
    }
  }

  // New variable to avoid recomputing in nested loops
  auto& phi_trace1 =
      get<::Tags::Tempi<0, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t i = 0; i < VolumeDim; ++i) {
    phi_trace1.get(i) = 0.;
    for (size_t j = 0; j < VolumeDim; ++j) {
      phi_trace1.get(i) += phi_loloup.get(i, j, j);
    }
  }
  auto& phi_trace2 =
      get<::Tags::Tempi<1, VolumeDim, Frame, DataType>>(local_buffer);
  for (size_t i = 0; i < VolumeDim; ++i) {
    phi_trace2.get(i) = 0.;
    for (size_t j = 0; j < VolumeDim; ++j) {
      phi_trace2.get(i) += phi_uplolo.get(j, j, i);
    }
  }

  auto& phi_trace1_up =
      get<::Tags::TempI<0, VolumeDim, Frame, DataType>>(local_buffer);
  auto& phi_trace2_up =
      get<::Tags::TempI<1, VolumeDim, Frame, DataType>>(local_buffer);

  raise_or_lower_index(make_not_null(&phi_trace1_up), phi_trace1,
                       inverse_spatial_metric);
  raise_or_lower_index(make_not_null(&phi_trace2_up), phi_trace2,
                       inverse_spatial_metric);

  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // Symmetry
      ricci->get(i, j) = 0.;
      for (size_t p = 0; p < VolumeDim; ++p) {
        ricci->get(i, j) +=
            0.5 *
            (phi.get(p, 1 + i, 1 + j) - phi.get(i, 1 + j, 1 + p) -
             phi.get(j, 1 + i, 1 + p)) *
            (2.0 * phi_trace2_up.get(p) - phi_trace1_up.get(p));
        for (size_t q = 0; q < VolumeDim; ++q) {
          ricci->get(i, j) +=
              0.25 * inverse_spatial_metric.get(p, q) *
                  (deriv_phi.get(j, q, 1 + p, 1 + i) +
                   deriv_phi.get(i, q, 1 + p, 1 + j) -
                   deriv_phi.get(j, i, 1 + p, 1 + q) -
                   deriv_phi.get(i, j, 1 + p, 1 + q) +
                   deriv_phi.get(p, i, 1 + q, 1 + j) +
                   deriv_phi.get(p, j, 1 + q, 1 + i) -
                   2.0 * deriv_phi.get(q, p, 1 + i, 1 + j)) +
              phi_loloup.get(i, p, q) * phi_loloup.get(j, q, p) +
              2.0 * phi_uplolo.get(p, i, q) * phi_loloup.get(p, j, q) -
              2.0 * phi_uplolo.get(p, q, i) * phi_uplolo.get(q, p, j);
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
/// \endcond
