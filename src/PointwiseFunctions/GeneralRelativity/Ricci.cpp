// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void ricci_tensor(
    const gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame, Index>*> result,
    const tnsr::Abb<DataType, SpatialDim, Frame, Index>& christoffel_2nd_kind,
    const tnsr::aBcc<DataType, SpatialDim, Frame, Index>&
        d_christoffel_2nd_kind) {
  destructive_resize_components(result,
                                get_size(get<0, 0, 0>(christoffel_2nd_kind)));
  for (auto& component : *result) {
    component = 0.0;
  }
  const auto dimensionality = index_dim<0>(*result);
  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = i; j < dimensionality; ++j) {
      for (size_t m = 0; m < dimensionality; ++m) {
        result->get(i, j) += d_christoffel_2nd_kind.get(m, m, i, j) -
                             0.5 * (d_christoffel_2nd_kind.get(i, m, m, j) +
                                    d_christoffel_2nd_kind.get(j, m, m, i));

        for (size_t n = 0; n < dimensionality; ++n) {
          result->get(i, j) += christoffel_2nd_kind.get(m, i, j) *
                                   christoffel_2nd_kind.get(n, n, m) -
                               christoffel_2nd_kind.get(m, i, n) *
                                   christoffel_2nd_kind.get(n, m, j);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame, Index> ricci_tensor(
    const tnsr::Abb<DataType, SpatialDim, Frame, Index>& christoffel_2nd_kind,
    const tnsr::aBcc<DataType, SpatialDim, Frame, Index>&
        d_christoffel_2nd_kind) {
  tnsr::aa<DataType, SpatialDim, Frame, Index> result{};
  ricci_tensor(make_not_null(&result), christoffel_2nd_kind,
               d_christoffel_2nd_kind);
  return result;
}

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void ricci_scalar(
    const gsl::not_null<Scalar<DataType>*> ricci_scalar_result,
    const tnsr::aa<DataType, SpatialDim, Frame, Index>& ricci_tensor,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric) {
  destructive_resize_components(ricci_scalar_result,
                                get_size(get<0, 0>(inverse_metric)));
  *ricci_scalar_result =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_metric), 0.0);

  const auto dimensionality = index_dim<0>(ricci_tensor);

  auto ricci_up_down = make_with_value<tnsr::Ab<DataType, SpatialDim, Frame>>(
      get<0, 0>(inverse_metric), 0.0);
  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = 0; j < dimensionality; ++j) {
      for (size_t k = 0; k < dimensionality; ++k) {
        ricci_up_down.get(j, k) +=
            ricci_tensor.get(i, k) * inverse_metric.get(i, j);
      }
    }
  }
  for (size_t j = 0; j < dimensionality; ++j) {
    get(*ricci_scalar_result) += ricci_up_down.get(j, j);
  }
}

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
Scalar<DataType> ricci_scalar(
    const tnsr::aa<DataType, SpatialDim, Frame, Index>& ricci_tensor,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric) {
  Scalar<DataType> ricci_scalar_result{get<0, 0>(inverse_metric)};
  ricci_scalar(make_not_null(&ricci_scalar_result), ricci_tensor,
               inverse_metric);
  return ricci_scalar_result;
}
} // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                  \
  template void gr::ricci_tensor(                                             \
      const gsl::not_null<                                                    \
          tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>*>    \
          result,                                                             \
      const tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          christoffel_2nd_kind,                                               \
      const tnsr::aBcc<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_christoffel_2nd_kind);                                            \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>     \
  gr::ricci_tensor(                                                           \
      const tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          christoffel_2nd_kind,                                               \
      const tnsr::aBcc<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_christoffel_2nd_kind);                                            \
  template Scalar<DTYPE(data)> gr::ricci_scalar(                              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&   \
          ricci_tensor,                                                       \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&   \
          inverse_metric);                                                    \
  template void gr::ricci_scalar(                                             \
      const gsl::not_null<Scalar<DTYPE(data)>*> ricci_scalar_result,          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&   \
          ricci_tensor,                                                       \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&   \
          inverse_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial),
                        (IndexType::Spatial, IndexType::Spacetime))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE
