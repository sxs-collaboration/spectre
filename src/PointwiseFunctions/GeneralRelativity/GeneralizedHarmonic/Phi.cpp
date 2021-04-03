// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
void phi(const gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> phi,
         const Scalar<DataType>& lapse,
         const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
         const tnsr::I<DataType, SpatialDim, Frame>& shift,
         const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
         const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
         const tnsr::ijj<DataType, SpatialDim, Frame>&
             deriv_spatial_metric) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*phi)) != get_size(get(lapse)))) {
    *phi = tnsr::iaa<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t k = 0; k < SpatialDim; ++k) {
    phi->get(k, 0, 0) = -2. * get(lapse) * deriv_lapse.get(k);
    for (size_t m = 0; m < SpatialDim; ++m) {
      for (size_t n = 0; n < SpatialDim; ++n) {
        phi->get(k, 0, 0) +=
            deriv_spatial_metric.get(k, m, n) * shift.get(m) * shift.get(n) +
            2. * spatial_metric.get(m, n) * shift.get(m) *
                deriv_shift.get(k, n);
      }
    }

    for (size_t i = 0; i < SpatialDim; ++i) {
      phi->get(k, 0, i + 1) = 0.;
      for (size_t m = 0; m < SpatialDim; ++m) {
        phi->get(k, 0, i + 1) +=
            deriv_spatial_metric.get(k, m, i) * shift.get(m) +
            spatial_metric.get(m, i) * deriv_shift.get(k, m);
      }
      for (size_t j = i; j < SpatialDim; ++j) {
        phi->get(k, i + 1, j + 1) = deriv_spatial_metric.get(k, i, j);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept {
  tnsr::iaa<DataType, SpatialDim, Frame> var_phi{};
  GeneralizedHarmonic::phi<SpatialDim, Frame, DataType>(
      make_not_null(&var_phi), lapse, deriv_lapse, shift, deriv_shift,
      spatial_metric, deriv_spatial_metric);
  return var_phi;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void GeneralizedHarmonic::phi(                                  \
      const gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*> \
          var_phi,                                                         \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          deriv_spatial_metric) noexcept;                                  \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::phi(                                                \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          deriv_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
