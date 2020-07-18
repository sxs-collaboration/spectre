// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfDetSpatialMetric.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace GeneralizedHarmonic {
namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
struct D4gBuffer;

template <size_t SpatialDim, typename Frame>
struct D4gBuffer<SpatialDim, Frame, double> {
  explicit D4gBuffer(const size_t /*size*/) noexcept {}

  tnsr::ijj<double, SpatialDim, Frame> deriv_of_g{};
  Scalar<double> det_spatial_metric{};
};

template <size_t SpatialDim, typename Frame>
struct D4gBuffer<SpatialDim, Frame, DataVector> {
 private:
  // We make one giant allocation so that we don't thrash the heap.
  Variables<tmpl::list<::Tags::Tempijj<0, SpatialDim, Frame, DataVector>,
                       ::Tags::TempScalar<1, DataVector>>>
      buffer_;

 public:
  explicit D4gBuffer(const size_t size) noexcept
      : buffer_(size),
        deriv_of_g(
            get<::Tags::Tempijj<0, SpatialDim, Frame, DataVector>>(buffer_)),
        det_spatial_metric(get<::Tags::TempScalar<1, DataVector>>(buffer_)) {}

  tnsr::ijj<DataVector, SpatialDim, Frame>& deriv_of_g;
  Scalar<DataVector>& det_spatial_metric;
};
}  // namespace

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_det_spatial_metric(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*>
        d4_det_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_det_spatial_metric)) !=
               get_size(get(sqrt_det_spatial_metric)))) {
    *d4_det_spatial_metric = tnsr::a<DataType, SpatialDim, Frame>(
        get_size(get(sqrt_det_spatial_metric)));
  }
  auto& d4_g = *d4_det_spatial_metric;
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  D4gBuffer<SpatialDim, Frame, DataType> buffer(
      get_size(get(sqrt_det_spatial_metric)));
  deriv_spatial_metric<SpatialDim, Frame, DataType>(
      make_not_null(&buffer.deriv_of_g), phi);
  get(buffer.det_spatial_metric) = square(get(sqrt_det_spatial_metric));
  // \f$ \partial_0 g = g g^{jk} \partial_0 g_{jk}\f$
  get<0>(d4_g) = inverse_spatial_metric.get(0, 0) * dt_spatial_metric.get(0, 0);
  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      if (LIKELY(j != 0 or k != 0)) {
        get<0>(d4_g) +=
            inverse_spatial_metric.get(j, k) * dt_spatial_metric.get(j, k);
      }
    }
  }
  get<0>(d4_g) *= get(buffer.det_spatial_metric);
  // \f$ \partial_i g = g g^{jk} \partial_i g_{jk}\f$
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_g.get(i + 1) =
        inverse_spatial_metric.get(0, 0) * buffer.deriv_of_g.get(i, 0, 0);
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        if (LIKELY(j != 0 or k != 0)) {
          d4_g.get(i + 1) +=
              inverse_spatial_metric.get(j, k) * buffer.deriv_of_g.get(i, j, k);
        }
      }
    }
    d4_g.get(i + 1) *= get(buffer.det_spatial_metric);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_det_spatial_metric(
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_det_spatial_metric{};
  GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric<SpatialDim, Frame,
                                                             DataType>(
      make_not_null(&d4_det_spatial_metric), sqrt_det_spatial_metric,
      inverse_spatial_metric, dt_spatial_metric, phi);
  return d4_det_spatial_metric;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric(   \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          d4_det_spatial_metric,                                              \
      const Scalar<DTYPE(data)>& det_spatial_metric,                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric(                 \
      const Scalar<DTYPE(data)>& det_spatial_metric,                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
