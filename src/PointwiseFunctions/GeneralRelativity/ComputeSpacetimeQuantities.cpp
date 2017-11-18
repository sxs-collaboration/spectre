// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t Dim, typename Fr, typename DataType>
tnsr::aa<DataType, Dim, Fr> compute_spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric) noexcept {
  tnsr::aa<DataType, Dim, Fr> spacetime_metric{
      make_with_value<DataType>(lapse.get(), 0.)};

  get<0, 0>(spacetime_metric) = -lapse.get() * lapse.get();

  for (size_t m = 0; m < Dim; ++m) {
    get<0, 0>(spacetime_metric) +=
        spatial_metric.get(m, m) * shift.get(m) * shift.get(m);
    for (size_t n = 0; n < m; ++n) {
      get<0, 0>(spacetime_metric) +=
          2 * spatial_metric.get(m, n) * shift.get(m) * shift.get(n);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t m = 0; m < Dim; ++m) {
      spacetime_metric.get(0, i + 1) += spatial_metric.get(m, i) * shift.get(m);
    }
    for (size_t j = i; j < Dim; ++j) {
      spacetime_metric.get(i + 1, j + 1) = spatial_metric.get(i, j);
    }
  }
  return spacetime_metric;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                            \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                \
  compute_spacetime_metric(                                             \
      const Scalar<DTYPE(data)>& lapse,                                 \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&              \
          spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
