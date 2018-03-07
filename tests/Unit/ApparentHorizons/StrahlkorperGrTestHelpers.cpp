// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/ApparentHorizons/StrahlkorperGrTestHelpers.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace TestHelpers {
namespace Schwarzschild {

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_ricci(
    const tnsr::I<DataType, SpatialDim, Frame>& x,
    const double& mass) noexcept {
  auto ricci = make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(x, 0.);

  constexpr auto dimensionality = index_dim<0>(ricci);

  const DataType r = get(magnitude(x));

  for (size_t i = 0; i < dimensionality; ++i) {
    for (size_t j = i; j < dimensionality; ++j) {
      ricci.get(i, j) -= (8.0 * mass + 3.0 * r) * x.get(i) * x.get(j);
      if (i == j) {
        ricci.get(i, j) += square(r) * (4.0 * mass + r);
      }
      ricci.get(i, j) *= mass;
      ricci.get(i, j) /= pow<4>(r) * square(2.0 * mass + r);
    }
  }

  return ricci;
}
}  // namespace Schwarzschild

namespace Minkowski {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature_sphere(
    const tnsr::I<DataType, SpatialDim, Frame>& x) noexcept {
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(x, 0.);

  constexpr auto dimensionality = index_dim<0>(extrinsic_curvature);

  const DataType one_over_r = 1.0 / get(magnitude(x));

  for (size_t i = 0; i < dimensionality; ++i) {
    extrinsic_curvature.get(i, i) += 1.0;
    for (size_t j = i; j < dimensionality; ++j) {
      extrinsic_curvature.get(i, j) -= x.get(i) * x.get(j) * square(one_over_r);
      extrinsic_curvature.get(i, j) *= one_over_r;
    }
  }

  return extrinsic_curvature;
}
}  // namespace Minkowski
}  // namespace TestHelpers

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                 \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>     \
  TestHelpers::Schwarzschild::spatial_ricci(                 \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x, \
      const double& mass) noexcept;                          \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>     \
  TestHelpers::Minkowski::extrinsic_curvature_sphere(        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INDEXTYPE
#undef INSTANTIATE
