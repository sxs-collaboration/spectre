// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"

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
void time_deriv_of_lapse(
    const gsl::not_null<Scalar<DataType>*> dt_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  if (UNLIKELY(get_size(get(*dt_lapse)) != get_size(get(lapse)))) {
    *dt_lapse = Scalar<DataType>(get_size(get(lapse)));
  }
  get(*dt_lapse) =
      get(lapse) * get<0, 0>(pi) * square(get<0>(spacetime_unit_normal));
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      // first term
      if (LIKELY(a != 0 or b != 0)) {
        get(*dt_lapse) += get(lapse) * pi.get(a, b) *
                          spacetime_unit_normal.get(a) *
                          spacetime_unit_normal.get(b);
      }
      // second term
      for (size_t i = 0; i < SpatialDim; ++i) {
        get(*dt_lapse) -= shift.get(i) * phi.get(i, a, b) *
                          spacetime_unit_normal.get(a) *
                          spacetime_unit_normal.get(b);
      }
    }
  }
  get(*dt_lapse) *= 0.5 * get(lapse);
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  Scalar<DataType> dt_lapse{};
  GeneralizedHarmonic::time_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal, phi, pi);
  return dt_lapse;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template void GeneralizedHarmonic::time_deriv_of_lapse(                \
      const gsl::not_null<Scalar<DTYPE(data)>*> dt_lapse,                \
      const Scalar<DTYPE(data)>& lapse,                                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_unit_normal,                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,         \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi);          \
  template Scalar<DTYPE(data)> GeneralizedHarmonic::time_deriv_of_lapse( \
      const Scalar<DTYPE(data)>& lapse,                                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_unit_normal,                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,         \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
