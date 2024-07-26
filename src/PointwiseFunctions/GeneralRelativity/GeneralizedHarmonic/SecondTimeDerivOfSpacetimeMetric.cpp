// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SecondTimeDerivOfSpacetimeMetric.hpp"

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {

template <typename DataType, size_t SpatialDim, typename Frame>
void second_time_deriv_of_spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> d2t2_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& dt_phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::aa<DataType, SpatialDim, Frame>& dt_pi) {
  if (UNLIKELY(get_size(get<0, 0>(*d2t2_spacetime_metric)) !=
               get_size(get<0, 0>(pi)))) {
    *d2t2_spacetime_metric =
        tnsr::aa<DataType, SpatialDim, Frame>(get_size(get<0, 0>(pi)));
  }
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = a; b < SpatialDim + 1; ++b) {
      d2t2_spacetime_metric->get(a, b) = -get(dt_lapse) * pi.get(a, b);
      d2t2_spacetime_metric->get(a, b) -= get(lapse) * dt_pi.get(a, b);
      for (size_t i = 0; i < SpatialDim; ++i) {
        d2t2_spacetime_metric->get(a, b) += dt_shift.get(i) * phi.get(i, a, b);
        d2t2_spacetime_metric->get(a, b) += shift.get(i) * dt_phi.get(i, a, b);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aa<DataType, SpatialDim, Frame> second_time_deriv_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& dt_phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::aa<DataType, SpatialDim, Frame>& dt_pi) {
  tnsr::aa<DataType, SpatialDim, Frame> d2t2_spacetime_metric{};
  gh::second_time_deriv_of_spacetime_metric(
      make_not_null(&d2t2_spacetime_metric), lapse, dt_lapse, shift, dt_shift,
      phi, dt_phi, pi, dt_pi);
  return d2t2_spacetime_metric;
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template void gh::second_time_deriv_of_spacetime_metric(                   \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>    \
          d2t2_spacetime_metric,                                             \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& dt_phi,          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& dt_pi);           \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                     \
  gh::second_time_deriv_of_spacetime_metric(                                 \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse, \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& dt_phi,          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& dt_pi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
