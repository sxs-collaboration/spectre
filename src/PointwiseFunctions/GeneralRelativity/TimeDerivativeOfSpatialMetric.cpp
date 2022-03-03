// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature) {
  destructive_resize_components(dt_spatial_metric, get_size(get(lapse)));
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      dt_spatial_metric->get(i, j) =
          -2. * get(lapse) * extrinsic_curvature.get(i, j);
      for (size_t k = 0; k < SpatialDim; ++k) {
        dt_spatial_metric->get(i, j) +=
            shift.get(k) * deriv_spatial_metric.get(k, i, j) +
            spatial_metric.get(k, i) * deriv_shift.get(j, k) +
            spatial_metric.get(k, j) * deriv_shift.get(i, k);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> time_derivative_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature) {
  tnsr::ii<DataType, SpatialDim, Frame> dt_spatial_metric{get_size(get(lapse))};
  time_derivative_of_spatial_metric(make_not_null(&dt_spatial_metric), lapse,
                                    shift, deriv_shift, spatial_metric,
                                    deriv_spatial_metric, extrinsic_curvature);
  return dt_spatial_metric;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                   \
  gr::time_derivative_of_spatial_metric(                                   \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          deriv_spatial_metric,                                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          extrinsic_curvature);                                            \
  template void gr::time_derivative_of_spatial_metric(                     \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>  \
          dt_spatial_metric,                                               \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          deriv_spatial_metric,                                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          extrinsic_curvature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
