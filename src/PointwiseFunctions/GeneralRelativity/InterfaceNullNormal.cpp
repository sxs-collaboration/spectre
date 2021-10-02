// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::a<DataType, VolumeDim, Frame> interface_null_normal(
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form,
    const double sign) {
  ASSERT((sign == 1.) or (sign == -1.),
         "Calculation of interface null normal accepts only +1/-1 to indicate "
         "whether the outgoing/incoming normal is needed.");
  tnsr::a<DataType, VolumeDim, Frame> null_one_form(
      get_size(get<0>(spacetime_normal_one_form)));
  interface_null_normal(make_not_null(&null_one_form),
                        spacetime_normal_one_form,
                        interface_unit_normal_one_form, sign);
  return null_one_form;
}

template <size_t VolumeDim, typename Frame, typename DataType>
void interface_null_normal(
    gsl::not_null<tnsr::a<DataType, VolumeDim, Frame>*> null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form,
    const double sign) {
  ASSERT((sign == 1.) or (sign == -1.),
         "Calculation of interface null normal accepts only +1/-1 to indicate "
         "whether the outgoing/incoming normal is needed.");
  destructive_resize_components(null_one_form,
                                get_size(get<0>(spacetime_normal_one_form)));

  const double one_by_sqrt_2 = 1. / sqrt(2.);
  get<0>(*null_one_form) = one_by_sqrt_2 * get<0>(spacetime_normal_one_form);
  for (size_t a = 1; a < VolumeDim + 1; ++a) {
    null_one_form->get(a) =
        one_by_sqrt_2 * (spacetime_normal_one_form.get(a) +
                         sign * interface_unit_normal_one_form.get(a - 1));
  }
}

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::A<DataType, VolumeDim, Frame> interface_null_normal(
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    const double sign) {
  ASSERT((sign == 1.) or (sign == -1.),
         "Calculation of interface null normal accepts only +1/-1 to indicate "
         "whether the outgoing/incoming normal is needed.");
  tnsr::A<DataType, VolumeDim, Frame> null_vector(
      get_size(get<0>(spacetime_normal_vector)));
  interface_null_normal(make_not_null(&null_vector), spacetime_normal_vector,
                        interface_unit_normal_vector, sign);
  return null_vector;
}

template <size_t VolumeDim, typename Frame, typename DataType>
void interface_null_normal(
    gsl::not_null<tnsr::A<DataType, VolumeDim, Frame>*> null_vector,
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    const double sign) {
  ASSERT((sign == 1.) or (sign == -1.),
         "Calculation of interface null normal accepts only +1/-1 to indicate "
         "whether the outgoing/incoming normal is needed.");
  destructive_resize_components(null_vector,
                                get_size(get<0>(spacetime_normal_vector)));

  const double one_by_sqrt_2 = 1. / sqrt(2.);
  get<0>(*null_vector) = one_by_sqrt_2 * get<0>(spacetime_normal_vector);
  for (size_t a = 1; a < VolumeDim + 1; ++a) {
    null_vector->get(a) =
        one_by_sqrt_2 * (spacetime_normal_vector.get(a) +
                         sign * interface_unit_normal_vector.get(a - 1));
  }
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                             \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                  \
  gr::interface_null_normal(                                             \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_normal_one_form,                                     \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                \
          interface_unit_normal_one_form,                                \
      const double sign);                                                \
  template void gr::interface_null_normal(                               \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*> \
          null_one_form,                                                 \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_normal_one_form,                                     \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                \
          interface_unit_normal_one_form,                                \
      const double sign);                                                \
  template tnsr::A<DTYPE(data), DIM(data), FRAME(data)>                  \
  gr::interface_null_normal(                                             \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_normal_vector,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                \
          interface_unit_normal_vector,                                  \
      const double sign);                                                \
  template void gr::interface_null_normal(                               \
      const gsl::not_null<tnsr::A<DTYPE(data), DIM(data), FRAME(data)>*> \
          null_vector,                                                   \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                \
          spacetime_normal_vector,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                \
          interface_unit_normal_vector,                                  \
      const double sign);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
