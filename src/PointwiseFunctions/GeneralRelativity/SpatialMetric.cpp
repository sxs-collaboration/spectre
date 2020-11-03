// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> local_spatial_metric{
      get_size(get<0, 0>(spacetime_metric))};
  spatial_metric(make_not_null(&local_spatial_metric), spacetime_metric);
  return local_spatial_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  destructive_resize_components(spatial_metric,
                                get_size(get<0, 0>(spacetime_metric)));
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric->get(i, j) = spacetime_metric.get(i + 1, j + 1);
    }
  }
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)> gr::spatial_metric( \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spacetime_metric) noexcept;                                        \
  template void gr::spatial_metric(                                          \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>    \
          spatial_metric,                                                    \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spacetime_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
