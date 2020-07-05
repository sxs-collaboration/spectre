// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"

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
Scalar<DataType> lapse(
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  Scalar<DataType> local_lapse{get_size(get<0, 0>(spacetime_metric))};
  lapse(make_not_null(&local_lapse), shift, spacetime_metric);
  return local_lapse;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void lapse(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  destructive_resize_components(lapse, get_size(get<0, 0>(spacetime_metric)));
  get(*lapse) = -get<0, 0>(spacetime_metric);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get(*lapse) += shift.get(i) * spacetime_metric.get(i + 1, 0);
  }
  get(*lapse) = sqrt(get(*lapse));
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                     \
  template Scalar<DTYPE(data)> gr::lapse(                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift, \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&       \
          spacetime_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
