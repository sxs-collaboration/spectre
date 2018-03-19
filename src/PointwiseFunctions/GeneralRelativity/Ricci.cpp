// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame, Index> ricci_tensor(
    const tnsr::Abb<DataType, SpatialDim, Frame, Index>& christoffel_2nd_kind,
    const tnsr::aBcc<DataType, SpatialDim, Frame, Index>&
        d_christoffel_2nd_kind) noexcept {
  auto ricci = make_with_value<tnsr::aa<DataType, SpatialDim, Frame, Index>>(
      christoffel_2nd_kind, 0.);
  constexpr auto dimensionality = index_dim<0>(ricci);
  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = i; j < dimensionality; ++j) {
      for (size_t m = 0; m < dimensionality; ++m) {
        ricci.get(i, j) += d_christoffel_2nd_kind.get(m, m, i, j) -
                           0.5 * (d_christoffel_2nd_kind.get(i, m, m, j) +
                                  d_christoffel_2nd_kind.get(j, m, m, i));

        for (size_t n = 0; n < dimensionality; ++n) {
          ricci.get(i, j) += christoffel_2nd_kind.get(m, i, j) *
                                 christoffel_2nd_kind.get(n, n, m) -
                             christoffel_2nd_kind.get(m, i, n) *
                                 christoffel_2nd_kind.get(n, m, j);
        }
      }
    }
  }
  return ricci;
}
} // namespace gr

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>     \
  gr::ricci_tensor(                                                           \
      const tnsr::Abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          christoffel_2nd_kind,                                               \
      const tnsr::aBcc<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          d_christoffel_2nd_kind) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial),
                        (IndexType::Spatial, IndexType::Spacetime))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE
/// \endcond
