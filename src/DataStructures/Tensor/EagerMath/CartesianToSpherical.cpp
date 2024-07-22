// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/CartesianToSpherical.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename DataType, size_t Dim, typename CoordsFrame>
void cartesian_to_spherical(
    const gsl::not_null<tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>>*>
        result,
    const tnsr::I<DataType, Dim, CoordsFrame>& x) {
  if constexpr (Dim == 1) {
    get<0>(*result) = get<0>(x);
  } else if constexpr (Dim == 2) {
    get<0>(*result) = hypot(get<0>(x), get<1>(x));
    get<1>(*result) = atan2(get<1>(x), get<0>(x));
  } else if constexpr (Dim == 3) {
    get<0>(*result) =
        sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)));
    get<1>(*result) = atan2(hypot(get<0>(x), get<1>(x)), get<2>(x));
    get<2>(*result) = atan2(get<1>(x), get<0>(x));
  }
}

template <typename DataType, size_t Dim, typename CoordsFrame>
tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>> cartesian_to_spherical(
    const tnsr::I<DataType, Dim, CoordsFrame>& x) {
  tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>> result{};
  cartesian_to_spherical(make_not_null(&result), x);
  return result;
}

template <typename DataType, size_t Dim, typename CoordsFrame>
void spherical_to_cartesian(
    const gsl::not_null<tnsr::I<DataType, Dim, CoordsFrame>*> result,
    const tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>>& x) {
  const auto& r = get<0>(x);
  if constexpr (Dim == 1) {
    get<0>(*result) = r;
  } else if constexpr (Dim == 2) {
    const auto& phi = get<1>(x);
    get<0>(*result) = r * cos(phi);
    get<1>(*result) = r * sin(phi);
  } else if constexpr (Dim == 3) {
    const auto& theta = get<1>(x);
    const auto& phi = get<2>(x);
    get<0>(*result) = r * sin(theta) * cos(phi);
    get<1>(*result) = r * sin(theta) * sin(phi);
    get<2>(*result) = r * cos(theta);
  }
}

template <typename DataType, size_t Dim, typename CoordsFrame>
tnsr::I<DataType, Dim, CoordsFrame> spherical_to_cartesian(
    const tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>>& x) {
  tnsr::I<DataType, Dim, CoordsFrame> result{};
  spherical_to_cartesian(make_not_null(&result), x);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template void cartesian_to_spherical(                                     \
      gsl::not_null<                                                        \
          tnsr::I<DTYPE(data), DIM(data), Frame::Spherical<FRAME(data)>>*>  \
          result,                                                           \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x);               \
  template tnsr::I<DTYPE(data), DIM(data), Frame::Spherical<FRAME(data)>>   \
  cartesian_to_spherical(                                                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x);               \
  template void spherical_to_cartesian(                                     \
      gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*> result,  \
      const tnsr::I<DTYPE(data), DIM(data), Frame::Spherical<FRAME(data)>>& \
          x);                                                               \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)>                     \
  spherical_to_cartesian(const tnsr::I<DTYPE(data), DIM(data),              \
                                       Frame::Spherical<FRAME(data)>>& x);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE
