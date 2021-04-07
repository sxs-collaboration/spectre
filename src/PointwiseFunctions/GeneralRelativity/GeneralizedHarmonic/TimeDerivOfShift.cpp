// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_shift(
    const gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> dt_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  if (UNLIKELY(get_size(get<0>(*dt_shift)) != get_size(get(lapse)))) {
    *dt_shift = tnsr::I<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_shift->get(i) = -get(lapse) * pi.get(1, 0) *
                       spacetime_unit_normal.get(0) *
                       inverse_spatial_metric.get(i, 0);
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        if (a != 0 or j != 0) {
          dt_shift->get(i) -= get(lapse) * pi.get(j + 1, a) *
                              spacetime_unit_normal.get(a) *
                              inverse_spatial_metric.get(i, j);
        }
        for (size_t k = 0; k < SpatialDim; ++k) {
          dt_shift->get(i) += shift.get(j) * spacetime_unit_normal.get(a) *
                              phi.get(j, k + 1, a) *
                              inverse_spatial_metric.get(i, k);
        }
      }
    }
    dt_shift->get(i) *= get(lapse);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> time_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  tnsr::I<DataType, SpatialDim, Frame> dt_shift{};
  GeneralizedHarmonic::time_deriv_of_shift<SpatialDim, Frame, DataType>(
      make_not_null(&dt_shift), lapse, shift, inverse_spatial_metric,
      spacetime_unit_normal, phi, pi);
  return dt_shift;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template void GeneralizedHarmonic::time_deriv_of_shift(                \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*> \
          dt_shift,                                                      \
      const Scalar<DTYPE(data)>& lapse,                                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,         \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&               \
          inverse_spatial_metric,                                        \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_unit_normal,                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,         \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept; \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::time_deriv_of_shift(                              \
      const Scalar<DTYPE(data)>& lapse,                                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,         \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&               \
          inverse_spatial_metric,                                        \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_unit_normal,                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,         \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
