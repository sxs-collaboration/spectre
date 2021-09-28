// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"

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
void spatial_deriv_of_lapse(
    const gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> deriv_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  if (UNLIKELY(get_size(get<0>(*deriv_lapse)) != get_size(get(lapse)))) {
    *deriv_lapse = tnsr::i<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_lapse->get(i) =
        phi.get(i, 0, 0) * square(get<0>(spacetime_unit_normal));
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        if (LIKELY(a != 0 or b != 0)) {
          deriv_lapse->get(i) += phi.get(i, a, b) *
                                 spacetime_unit_normal.get(a) *
                                 spacetime_unit_normal.get(b);
        }
      }
    }
    deriv_lapse->get(i) *= -0.5 * get(lapse);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> spatial_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  tnsr::i<DataType, SpatialDim, Frame> deriv_lapse{};
  GeneralizedHarmonic::spatial_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&deriv_lapse), lapse, spacetime_unit_normal, phi);
  return deriv_lapse;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template void GeneralizedHarmonic::spatial_deriv_of_lapse(             \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*> \
          deriv_lapse,                                                   \
      const Scalar<DTYPE(data)>& lapse,                                  \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_unit_normal,                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);        \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::spatial_deriv_of_lapse(                           \
      const Scalar<DTYPE(data)>& lapse,                                  \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_unit_normal,                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
