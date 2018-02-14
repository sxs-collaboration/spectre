// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/ApparentHorizons/StrahlkorperGrTestHelpers.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame, Index> make_spatial_ricci_schwarzschild(
    const tnsr::A<DataType, SpatialDim, Frame, Index>& x,
    const double& mass) noexcept {
  auto r = make_with_value<Scalar<DataType>>(x, 0.);
  auto ricci =
      make_with_value<tnsr::aa<DataType, SpatialDim, Frame, Index>>(x, 0.);

  constexpr auto dimensionality = index_dim<0>(ricci);

  r.get() = magnitude(x);

  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = i; j < dimensionality; ++j) {
      ricci.get(i, j) -= (8.0 * mass + 3.0 * r.get()) * x.get(i) * x.get(j);
      if (i == j) {
        ricci.get(i, j) += r.get() * r.get() * (4.0 * mass + r.get());
      }
      ricci.get(i, j) *= mass;
      ricci.get(i, j) /= r.get() * r.get() * r.get() * r.get() *
                         (2.0 * mass + r.get()) * (2.0 * mass + r.get());
    }
  }

  return ricci;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>     \
  make_spatial_ricci_schwarzschild(                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& x, \
      const double& mass) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial), (IndexType::Spatial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE
