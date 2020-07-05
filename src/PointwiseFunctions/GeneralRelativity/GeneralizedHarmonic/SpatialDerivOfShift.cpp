// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace GeneralizedHarmonic {

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_shift(
    const gsl::not_null<tnsr::iJ<DataType, SpatialDim, Frame>*> deriv_shift,
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*deriv_shift)) != get_size(get(lapse)))) {
    *deriv_shift = tnsr::iJ<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_shift->get(i, j) =
          (inverse_spacetime_metric.get(j + 1, 0) +
           spacetime_unit_normal.get(j + 1) * spacetime_unit_normal.get(0)) *
          spacetime_unit_normal.get(0) * phi.get(i, 0, 0);
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        for (size_t b = 0; b < SpatialDim + 1; ++b) {
          if (a != 0 or b != 0) {
            deriv_shift->get(i, j) += (inverse_spacetime_metric.get(j + 1, a) +
                                       spacetime_unit_normal.get(j + 1) *
                                           spacetime_unit_normal.get(a)) *
                                      spacetime_unit_normal.get(b) *
                                      phi.get(i, a, b);
          }
        }
      }
      deriv_shift->get(i, j) *= get(lapse);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iJ<DataType, SpatialDim, Frame> spatial_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::iJ<DataType, SpatialDim, Frame> deriv_shift{};
  GeneralizedHarmonic::spatial_deriv_of_shift<SpatialDim, Frame, DataType>(
      make_not_null(&deriv_shift), lapse, inverse_spacetime_metric,
      spacetime_unit_normal, phi);
  return deriv_shift;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void GeneralizedHarmonic::spatial_deriv_of_shift(               \
      const gsl::not_null<tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>*>  \
          deriv_shift,                                                     \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spacetime_metric,                                        \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_unit_normal,                                           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept; \
  template tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>                   \
  GeneralizedHarmonic::spatial_deriv_of_shift(                             \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spacetime_metric,                                        \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_unit_normal,                                           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
