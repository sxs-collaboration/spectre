// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame, Index> compute_christoffel_first_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) {
  tnsr::abb<DataType, SpatialDim, Frame, Index> christoffel{};
  constexpr auto dimensionality =
      tnsr::abb<DataType, SpatialDim, Frame, Index>::template index_dim<0>();
  for (size_t k = 0; k < dimensionality; ++k) {
    for (size_t i = 0; i < dimensionality; ++i) {
      for (size_t j = i; j < dimensionality; ++j) {
        christoffel.get(k, i, j) =
            0.5 * (d_metric.get(i, j, k) + d_metric.get(j, i, k) -
                   d_metric.get(k, i, j));
      }
    }
  }
  return christoffel;
}

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>   \
  compute_christoffel_first_kind(                                            \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial),
                        (IndexType::Spatial, IndexType::Spacetime))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE
/// \endcond
