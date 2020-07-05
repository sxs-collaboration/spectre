// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace gr {
template <size_t Dim, typename Frame, typename DataType>
tnsr::AA<DataType, Dim, Frame> inverse_spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) noexcept {
  tnsr::AA<DataType, Dim, Frame> inv_spacetime_metric{};
  inverse_spacetime_metric(make_not_null(&inv_spacetime_metric), lapse, shift,
                           inverse_spatial_metric);
  return inv_spacetime_metric;
}

template <size_t Dim, typename Frame, typename DataType>
void inverse_spacetime_metric(
    const gsl::not_null<tnsr::AA<DataType, Dim, Frame>*>
        inverse_spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) noexcept {
  get<0, 0>(*inverse_spacetime_metric) = -1.0 / (get(lapse) * get(lapse));

  const auto& minus_one_over_lapse_sqrd = get<0, 0>(*inverse_spacetime_metric);

  for (size_t i = 0; i < Dim; ++i) {
    inverse_spacetime_metric->get(0, i + 1) =
        -shift.get(i) * minus_one_over_lapse_sqrd;
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inverse_spacetime_metric->get(i + 1, j + 1) =
          inverse_spatial_metric.get(i, j) +
          shift.get(i) * shift.get(j) * minus_one_over_lapse_sqrd;
    }
  }
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void gr::inverse_spacetime_metric(                             \
      const gsl::not_null<tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>*> \
          inv_spacetime_metric,                                           \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_spatial_metric) noexcept;                               \
  template tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>                  \
  gr::inverse_spacetime_metric(                                           \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
