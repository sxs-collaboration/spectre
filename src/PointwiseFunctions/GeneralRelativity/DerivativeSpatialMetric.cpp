// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
template <size_t Dim, typename Frame, typename DataType>
void deriv_inverse_spatial_metric(
    const gsl::not_null<tnsr::iJJ<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric) noexcept {
  destructive_resize_components(result,
                                get_size(get<0, 0>(inverse_spatial_metric)));
  for (auto& component : *result) {
    component = 0.0;
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      for (size_t k = 0; k < Dim; ++k) {
        for (size_t m = 0; m < Dim; ++m) {
          for (size_t n = 0; n < Dim; ++n) {
            (*result).get(k, i, j) -= inverse_spatial_metric.get(i, n) *
                                      inverse_spatial_metric.get(m, j) *
                                      d_spatial_metric.get(k, n, m);
          }
        }
      }
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJJ<DataType, Dim, Frame> deriv_inverse_spatial_metric(
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric) noexcept {
  tnsr::iJJ<DataType, Dim, Frame> result{};
  deriv_inverse_spatial_metric(make_not_null(&result), inverse_spatial_metric,
                               d_spatial_metric);
  return result;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void gr::deriv_inverse_spatial_metric(                          \
      const gsl::not_null<tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                          \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_spatial_metric) noexcept;                                      \
  template tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>                  \
  gr::deriv_inverse_spatial_metric(                                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                          \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
