// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
void gauge_source(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> gauge_source_h,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>&
        trace_christoffel_last_indices) noexcept {
  destructive_resize_components(gauge_source_h, get_size(get(lapse)));
  for (auto& component : *gauge_source_h) {
    component = 0.0;
  }
  DataType one_over_lapse = 1.0 / get(lapse);

  // Temporary to avoid more nested loops.
  auto temp = dt_shift;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      temp.get(i) -= shift.get(k) * deriv_shift.get(k, i);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      gauge_source_h->get(i + 1) += spatial_metric.get(i, k) * temp.get(k);
    }
    gauge_source_h->get(i + 1) *= square(one_over_lapse);
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    gauge_source_h->get(i + 1) += one_over_lapse * deriv_lapse.get(i) -
                                  trace_christoffel_last_indices.get(i);
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    get<0>(*gauge_source_h) +=
        shift.get(i) *
        (gauge_source_h->get(i + 1) + deriv_lapse.get(i) * one_over_lapse);
  }
  get<0>(*gauge_source_h) -= one_over_lapse * get(dt_lapse) +
                             get(lapse) * get(trace_extrinsic_curvature);
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_source(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>&
        trace_christoffel_last_indices) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> gauge_source_h{};
  gauge_source(make_not_null(&gauge_source_h), lapse, dt_lapse, deriv_lapse,
               shift, dt_shift, deriv_shift, spatial_metric,
               trace_extrinsic_curvature, trace_christoffel_last_indices);
  return gauge_source_h;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template void GeneralizedHarmonic::gauge_source(                           \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>     \
          gauge_source_h,                                                    \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,          \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,      \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,   \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,                  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                    \
          trace_christoffel_last_indices) noexcept;                          \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::gauge_source(                                         \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,          \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,      \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,   \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,                  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                    \
          trace_christoffel_last_indices) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
