// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spacetime_metric(
    const gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*>
        dt_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept {
  destructive_resize_components(dt_spacetime_metric, get_size(get(dt_lapse)));
  get<0, 0>(*dt_spacetime_metric) = -2.0 * get(lapse) * get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      get<0, 0>(*dt_spacetime_metric) +=
          dt_spatial_metric.get(i, j) * shift.get(i) * shift.get(j) +
          2.0 * spatial_metric.get(i, j) * shift.get(i) * dt_shift.get(j);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_spacetime_metric->get(0, i + 1) =
        dt_spatial_metric.get(i, 0) * get<0>(shift) +
        spatial_metric.get(i, 0) * get<0>(dt_shift);
    for (size_t j = 1; j < SpatialDim; ++j) {
      dt_spacetime_metric->get(0, i + 1) +=
          dt_spatial_metric.get(i, j) * shift.get(j) +
          spatial_metric.get(i, j) * dt_shift.get(j);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      dt_spacetime_metric->get(i + 1, j + 1) = dt_spatial_metric.get(i, j);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> time_derivative_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept {
  tnsr::aa<DataType, SpatialDim, Frame> dt_spacetime_metric{
      get_size(get(lapse))};
  time_derivative_of_spacetime_metric(make_not_null(&dt_spacetime_metric),
                                      lapse, dt_lapse, shift, dt_shift,
                                      spatial_metric, dt_spatial_metric);
  return dt_spacetime_metric;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                     \
  gr::time_derivative_of_spacetime_metric(                                   \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                   \
          dt_spatial_metric) noexcept;                                       \
  template void gr::time_derivative_of_spacetime_metric(                     \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>    \
          dt_spacetime_metric,                                               \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,   \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                   \
          dt_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
