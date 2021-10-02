// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_one_form(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> normal_one_form,
    const Scalar<DataType>& lapse) {
  destructive_resize_components(normal_one_form, get_size(get(lapse)));
  for (auto& component : *normal_one_form) {
    component = 0.0;
  }
  get<0>(*normal_one_form) = -get(lapse);
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse) {
  tnsr::a<DataType, SpatialDim, Frame> normal_one_form{};
  spacetime_normal_one_form(make_not_null(&normal_one_form), lapse);
  return normal_one_form;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                  \
  gr::spacetime_normal_one_form(const Scalar<DTYPE(data)>& lapse);       \
  template void gr::spacetime_normal_one_form(                           \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*> \
          normal_one_form,                                               \
      const Scalar<DTYPE(data)>& lapse);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
