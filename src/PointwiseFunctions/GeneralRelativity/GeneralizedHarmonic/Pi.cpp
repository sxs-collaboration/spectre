// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
void pi(const gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> pi,
        const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
        const tnsr::I<DataType, SpatialDim, Frame>& shift,
        const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
        const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
        const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
        const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*pi)) != get_size(get(lapse)))) {
    *pi = tnsr::aa<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }

  get<0, 0>(*pi) = -2. * get(lapse) * get(dt_lapse);

  for (size_t m = 0; m < SpatialDim; ++m) {
    for (size_t n = 0; n < SpatialDim; ++n) {
      get<0, 0>(*pi) +=
          dt_spatial_metric.get(m, n) * shift.get(m) * shift.get(n) +
          2. * spatial_metric.get(m, n) * shift.get(m) * dt_shift.get(n);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    pi->get(0, i + 1) = 0.;
    for (size_t m = 0; m < SpatialDim; ++m) {
      pi->get(0, i + 1) += dt_spatial_metric.get(m, i) * shift.get(m) +
                           spatial_metric.get(m, i) * dt_shift.get(m);
    }
    for (size_t j = i; j < SpatialDim; ++j) {
      pi->get(i + 1, j + 1) = dt_spatial_metric.get(i, j);
    }
  }
  for (size_t mu = 0; mu < SpatialDim + 1; ++mu) {
    for (size_t nu = mu; nu < SpatialDim + 1; ++nu) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        pi->get(mu, nu) -= shift.get(i) * phi.get(i, mu, nu);
      }
      // Division by `lapse` here is somewhat more efficient (in Release mode)
      // than pre-computing `one_over_lapse` outside the loop for DataVectors
      // of `size` up to `50`. This is why we the next line is as it is.
      pi->get(mu, nu) /= -get(lapse);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::aa<DataType, SpatialDim, Frame> pi{};
  GeneralizedHarmonic::pi<SpatialDim, Frame, DataType>(
      make_not_null(&pi), lapse, dt_lapse, shift, dt_shift, spatial_metric,
      dt_spatial_metric, phi);
  return pi;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void GeneralizedHarmonic::pi(                                      \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>     \
          var_pi,                                                             \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::pi(                                                    \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
