// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void deriv_conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::iJkk<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::ijkk<DataType, Dim, Frame>& d_field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up) noexcept {
  destructive_resize_components(
      result, get_size(get<0, 0>(inverse_conformal_spatial_metric)));

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      for (size_t k = 0; k < Dim; ++k) {
        for (size_t m = 0; m < Dim; ++m) {
          (*result).get(k, m, i, j) =
              -2.0 * field_d_up.get(k, m, 0) *
                  (field_d.get(i, j, 0) + field_d.get(j, i, 0) -
                   field_d.get(0, i, j)) +
              0.5 * inverse_conformal_spatial_metric.get(m, 0) *
                  (d_field_d.get(k, i, j, 0) + d_field_d.get(i, k, j, 0) +
                   d_field_d.get(k, j, i, 0) + d_field_d.get(j, k, i, 0) -
                   d_field_d.get(k, 0, i, j) - d_field_d.get(0, k, i, j));
          for (size_t l = 1; l < Dim; ++l) {
            (*result).get(k, m, i, j) +=
                -2.0 * field_d_up.get(k, m, l) *
                    (field_d.get(i, j, l) + field_d.get(j, i, l) -
                     field_d.get(l, i, j)) +
                0.5 * inverse_conformal_spatial_metric.get(m, l) *
                    (d_field_d.get(k, i, j, l) + d_field_d.get(i, k, j, l) +
                     d_field_d.get(k, j, i, l) + d_field_d.get(j, k, i, l) -
                     d_field_d.get(k, l, i, j) - d_field_d.get(l, k, i, j));
          }
        }
      }
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJkk<DataType, Dim, Frame> deriv_conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::ijkk<DataType, Dim, Frame>& d_field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up) noexcept {
  tnsr::iJkk<DataType, Dim, Frame> result{};
  deriv_conformal_christoffel_second_kind(make_not_null(&result),
                                          inverse_conformal_spatial_metric,
                                          field_d, d_field_d, field_d_up);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template void Ccz4::deriv_conformal_christoffel_second_kind(              \
      const gsl::not_null<tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_conformal_spatial_metric,                                 \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,        \
      const tnsr::ijkk<DTYPE(data), DIM(data), FRAME(data)>& d_field_d,     \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>&                 \
          field_d_up) noexcept;                                             \
  template tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::deriv_conformal_christoffel_second_kind(                            \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                  \
          inverse_conformal_spatial_metric,                                 \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,        \
      const tnsr::ijkk<DTYPE(data), DIM(data), FRAME(data)>& d_field_d,     \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>&                 \
          field_d_up) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
