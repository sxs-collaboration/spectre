// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Christoffel.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d) noexcept {
  destructive_resize_components(
      result, get_size(get<0, 0>(inverse_conformal_spatial_metric)));

  ::TensorExpressions::evaluate<ti_K, ti_i, ti_j>(
      result, inverse_conformal_spatial_metric(ti_K, ti_L) *
                  (field_d(ti_i, ti_j, ti_l) + field_d(ti_j, ti_i, ti_l) -
                   field_d(ti_l, ti_i, ti_j)));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::Ijj<DataType, Dim, Frame> conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d) noexcept {
  tnsr::Ijj<DataType, Dim, Frame> result{};
  conformal_christoffel_second_kind(make_not_null(&result),
                                    inverse_conformal_spatial_metric, field_d);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template void Ccz4::conformal_christoffel_second_kind(                       \
      const gsl::not_null<tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>*>     \
          result,                                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_conformal_spatial_metric,                                    \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d) noexcept; \
  template tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>                      \
  Ccz4::conformal_christoffel_second_kind(                                     \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_conformal_spatial_metric,                                    \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
