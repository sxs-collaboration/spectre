// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> shift(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  tnsr::I<DataType, SpatialDim, Frame> local_shift{
      get_size(get<0, 0>(spacetime_metric))};
  shift(make_not_null(&local_shift), spacetime_metric, inverse_spatial_metric);
  return local_shift;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void shift(
    const gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  destructive_resize_components(shift, get_size(get<0, 0>(spacetime_metric)));
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift->get(i) =
        inverse_spatial_metric.get(i, 0) * get<1, 0>(spacetime_metric);
    for (size_t j = 1; j < SpatialDim; ++j) {
      shift->get(i) +=
          inverse_spatial_metric.get(i, j) * spacetime_metric.get(j + 1, 0);
    }
  }
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> gr::shift(           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric);                                           \
  template void gr::shift(                                                   \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>     \
          shift,                                                             \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
