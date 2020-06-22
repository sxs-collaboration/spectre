// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace gr {
template <size_t Dim, typename Frame, typename DataType>
void spacetime_metric(
    const gsl::not_null<tnsr::aa<DataType, Dim, Frame>*> spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*spacetime_metric)) !=
               get_size(get(lapse)))) {
    *spacetime_metric = tnsr::aa<DataType, Dim, Frame>(get_size(get(lapse)));
  }

  get<0, 0>(*spacetime_metric) = -square(get(lapse));

  for (size_t m = 0; m < Dim; ++m) {
    get<0, 0>(*spacetime_metric) +=
        spatial_metric.get(m, m) * square(shift.get(m));
    for (size_t n = 0; n < m; ++n) {
      get<0, 0>(*spacetime_metric) +=
          2. * spatial_metric.get(m, n) * shift.get(m) * shift.get(n);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    spacetime_metric->get(0, i + 1) = 0.;
    for (size_t m = 0; m < Dim; ++m) {
      spacetime_metric->get(0, i + 1) +=
          spatial_metric.get(m, i) * shift.get(m);
    }
    for (size_t j = i; j < Dim; ++j) {
      spacetime_metric->get(i + 1, j + 1) = spatial_metric.get(i, j);
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::aa<DataType, Dim, Frame> spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  tnsr::aa<DataType, Dim, Frame> spacetime_metric{};
  gr::spacetime_metric<Dim, Frame, DataType>(make_not_null(&spacetime_metric),
                                             lapse, shift, spatial_metric);
  return spacetime_metric;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template void gr::spacetime_metric(                                          \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>      \
          spacetime_metric,                                                    \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spatial_metric) noexcept;                                            \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)> gr::spacetime_metric( \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
