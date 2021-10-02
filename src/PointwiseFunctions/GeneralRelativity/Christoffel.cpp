// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void christoffel_first_kind(
    const gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame, Index>*>
        christoffel,
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) {
  constexpr auto dimensionality =
      Index == IndexType::Spatial ? SpatialDim : SpatialDim + 1;
  for (size_t k = 0; k < dimensionality; ++k) {
    for (size_t i = 0; i < dimensionality; ++i) {
      for (size_t j = i; j < dimensionality; ++j) {
        christoffel->get(k, i, j) =
            0.5 * (d_metric.get(i, j, k) + d_metric.get(j, i, k) -
                   d_metric.get(k, i, j));
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame, Index> christoffel_first_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) {
  auto christoffel =
      make_with_value<tnsr::abb<DataType, SpatialDim, Frame, Index>>(d_metric,
                                                                     0.);
  christoffel_first_kind(make_not_null(&christoffel), d_metric);
  return christoffel;
}

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Abb<DataType, SpatialDim, Frame, Index>*>
        christoffel,
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric) {
  destructive_resize_components(christoffel, get_size(*d_metric.begin()));
  constexpr auto dimensionality =
      Index == IndexType::Spatial ? SpatialDim : SpatialDim + 1;
  for (size_t d = 0; d < dimensionality; ++d) {
    for (size_t a = 0; a < dimensionality; ++a) {
      for (size_t b = a; b < dimensionality; ++b) {
        christoffel->get(d, a, b) =
            0.5 * inverse_metric.get(0, d) *
            (d_metric.get(a, b, 0) + d_metric.get(b, a, 0) -
             d_metric.get(0, a, b));
        for (size_t c = 1; c < dimensionality; ++c) {
          christoffel->get(d, a, b) +=
              0.5 * inverse_metric.get(c, d) *
              (d_metric.get(a, b, c) + d_metric.get(b, a, c) -
               d_metric.get(c, a, b));
        }
      }
    }
  }
}
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
auto christoffel_second_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric)
    -> tnsr::Abb<DataType, SpatialDim, Frame, Index> {
  tnsr::Abb<DataType, SpatialDim, Frame, Index> christoffel(
      get_size(get<0, 0, 0>(d_metric)));
  christoffel_second_kind(make_not_null(&christoffel), d_metric,
                          inverse_metric);
  return christoffel;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>   \
  gr::christoffel_first_kind(                                                \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric);                                                         \
  template void gr::christoffel_first_kind(                                  \
      const gsl::not_null<                                                   \
          tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>*>  \
          christoffel,                                                       \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric);                                                         \
  template tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>   \
  gr::christoffel_second_kind(                                               \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric,                                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          inverse_metric);                                                   \
  template void gr::christoffel_second_kind(                                 \
      const gsl::not_null<                                                   \
          tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>*>  \
          christoffel,                                                       \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric,                                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          inverse_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>,
                         Frame::Spherical<Frame::Grid>),
                        (IndexType::Spatial, IndexType::Spacetime))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE
