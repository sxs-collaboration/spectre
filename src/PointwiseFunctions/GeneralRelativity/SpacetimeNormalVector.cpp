// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept {
  tnsr::A<DataType, SpatialDim, Frame> local_spacetime_normal_vector{
      get_size(get(lapse))};
  spacetime_normal_vector(make_not_null(&local_spacetime_normal_vector), lapse,
                          shift);
  return local_spacetime_normal_vector;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_vector(
    const gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>
        spacetime_normal_vector,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept {
  destructive_resize_components(spacetime_normal_vector, get_size(get(lapse)));
  get<0>(*spacetime_normal_vector) = 1. / get(lapse);
  for (size_t i = 0; i < SpatialDim; i++) {
    spacetime_normal_vector->get(i + 1) =
        -shift.get(i) * get<0>(*spacetime_normal_vector);
  }
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                            \
  template tnsr::A<DTYPE(data), DIM(data), FRAME(data)> \
  gr::spacetime_normal_vector(                          \
      const Scalar<DTYPE(data)>& lapse,                 \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
