// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, SpatialDim, Frame>*> christoffel,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric) {
  destructive_resize_components(christoffel, get_size(*phi.begin()));
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t m = 0; m < SpatialDim; ++m) {
        christoffel->get(m, i, j) =
            0.5 * inv_metric.get(m, 0) *
            (phi.get(i, j + 1, 1) + phi.get(j, i + 1, 1) -
             phi.get(0, i + 1, j + 1));
        for (size_t k = 1; k < SpatialDim; ++k) {
          christoffel->get(m, i, j) +=
              0.5 * inv_metric.get(m, k) *
              (phi.get(i, j + 1, k + 1) + phi.get(j, i + 1, k + 1) -
               phi.get(k, i + 1, j + 1));
        }
      }
    }
  }
}
template <size_t SpatialDim, typename Frame, typename DataType>
auto christoffel_second_kind(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric)
    -> tnsr::Ijj<DataType, SpatialDim, Frame> {
  tnsr::Ijj<DataType, SpatialDim, Frame> christoffel(
      get_size(get<0, 0, 0>(phi)));
  christoffel_second_kind(make_not_null(&christoffel), phi, inv_metric);
  return christoffel;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void GeneralizedHarmonic::christoffel_second_kind(              \
      const gsl::not_null<tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>*> \
          christoffel,                                                     \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_metric);    \
  template tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::christoffel_second_kind(                            \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>,
                         Frame::Spherical<Frame::Grid>))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
