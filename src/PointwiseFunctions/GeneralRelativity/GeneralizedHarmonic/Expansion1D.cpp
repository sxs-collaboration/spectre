// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Expansion1D.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
template <typename DataType, typename Frame>
void expansion1D(const gsl::not_null<Scalar<DataType>*> expansion,
                 const tnsr::ii<DataType, 3, Frame>& spatial_metric,
                 const tnsr::ijj<DataType, 3, Frame>& deriv_spatial_metric,
                 const tnsr::ii<DataType, 3, Frame>& ext_curvature,
                 const tnsr::I<DataType, 3, Frame>& coords) {
  expansion->get() = (2.0 * spatial_metric.get(2, 2) +
                      get<0>(coords) * deriv_spatial_metric.get(0, 2, 2)) /
                         (get<0>(coords) * spatial_metric.get(2, 2) *
                          sqrt(spatial_metric.get(0, 0))) -
                     2.0 * ext_curvature.get(2, 2) / spatial_metric.get(2, 2);
}

template <typename DataType, typename Frame>
Scalar<DataType> expansion1D(
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const tnsr::ijj<DataType, 3, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, 3, Frame>& ext_curvature,
    const tnsr::I<DataType, 3, Frame>& coords) {
  Scalar<DataType> var_exp_1D{};
  gh::expansion1D<DataType, Frame>(make_not_null(&var_exp_1D), spatial_metric,
                                   deriv_spatial_metric, ext_curvature, coords);
  return var_exp_1D;
}
}  // namespace gh

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template void gh::expansion1D(                                          \
      const gsl::not_null<Scalar<DTYPE(data)>*> var_exp_1D,               \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_metric,        \
      const tnsr::ijj<DTYPE(data), 3, FRAME(data)>& deriv_spatial_metric, \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& ext_curvature,         \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& coords);                \
  template Scalar<DTYPE(data)> gh::expansion1D(                           \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_metric,        \
      const tnsr::ijj<DTYPE(data), 3, FRAME(data)>& deriv_spatial_metric, \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& ext_curvature,         \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
