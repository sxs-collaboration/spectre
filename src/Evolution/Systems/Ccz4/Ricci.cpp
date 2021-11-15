// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Ricci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> buffer,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>& d_conformal_christoffel_second_kind,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p) {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));
  destructive_resize_components(buffer,
                                get_size(get<0, 0>(conformal_spatial_metric)));

  // Add first terms of \partial_m \Gamma^m_{ij} and
  // -\partial_j \Gamma^m_{im}
  ::TensorExpressions::evaluate<ti_i, ti_j>(
      buffer, d_conformal_christoffel_second_kind(ti_m, ti_M, ti_i, ti_j) -
                  d_conformal_christoffel_second_kind(ti_j, ti_M, ti_i, ti_m));

  TensorExpressions::evaluate<ti_i, ti_j>(
      result,
      (*buffer)(ti_i, ti_j) +
          // Add terms of \partial_m \Gamma^m_{ij} and
          // -\partial_j \Gamma^m_{im} that have a coefficient of 2
          2.0 * ((field_d_up(ti_m, ti_M, ti_L) *
                  (conformal_spatial_metric(ti_j, ti_l) * field_p(ti_i) +
                   conformal_spatial_metric(ti_i, ti_l) * field_p(ti_j) -
                   conformal_spatial_metric(ti_i, ti_j) * field_p(ti_l))) -
                 inverse_conformal_spatial_metric(ti_M, ti_L) *
                     (field_d(ti_m, ti_j, ti_l) * field_p(ti_i) +
                      field_d(ti_m, ti_i, ti_l) * field_p(ti_j) -
                      field_d(ti_m, ti_i, ti_j) * field_p(ti_l)) -
                 (field_d_up(ti_j, ti_M, ti_L) *
                  (conformal_spatial_metric(ti_m, ti_l) * field_p(ti_i) +
                   conformal_spatial_metric(ti_i, ti_l) * field_p(ti_m) -
                   conformal_spatial_metric(ti_i, ti_m) * field_p(ti_l))) +
                 inverse_conformal_spatial_metric(ti_M, ti_L) *
                     (field_d(ti_j, ti_m, ti_l) * field_p(ti_i) +
                      field_d(ti_j, ti_i, ti_l) * field_p(ti_m) -
                      field_d(ti_j, ti_i, ti_m) * field_p(ti_l))) -
          // Add \partial_{(i} P_{j)} type terms
          0.5 * (inverse_conformal_spatial_metric(ti_M, ti_L) *
                 (conformal_spatial_metric(ti_j, ti_l) *
                      (d_field_p(ti_m, ti_i) + d_field_p(ti_i, ti_m)) +
                  conformal_spatial_metric(ti_i, ti_l) *
                      (d_field_p(ti_m, ti_j) + d_field_p(ti_j, ti_m)) -
                  conformal_spatial_metric(ti_i, ti_j) *
                      (d_field_p(ti_m, ti_l) + d_field_p(ti_l, ti_m)) -
                  conformal_spatial_metric(ti_m, ti_l) *
                      (d_field_p(ti_j, ti_i) + d_field_p(ti_i, ti_j)) -
                  conformal_spatial_metric(ti_i, ti_l) *
                      (d_field_p(ti_j, ti_m) + d_field_p(ti_m, ti_j)) +
                  conformal_spatial_metric(ti_i, ti_m) *
                      (d_field_p(ti_j, ti_l) + d_field_p(ti_l, ti_j)))) +
          // Add last two terms for R_{ij}
          christoffel_second_kind(ti_L, ti_i, ti_j) *
              christoffel_second_kind(ti_M, ti_l, ti_m) -
          christoffel_second_kind(ti_L, ti_i, ti_m) *
              christoffel_second_kind(ti_M, ti_l, ti_j));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::ii<DataType, Dim, Frame> spatial_ricci_tensor(
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>& d_conformal_christoffel_second_kind,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p) {
  tnsr::ii<DataType, Dim, Frame> result{};
  tnsr::ij<DataType, Dim, Frame> buffer{};
  spatial_ricci_tensor(
      make_not_null(&result), make_not_null(&buffer), christoffel_second_kind,
      d_conformal_christoffel_second_kind, conformal_spatial_metric,
      inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
      d_field_p);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void Ccz4::spatial_ricci_tensor(                               \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                         \
      const gsl::not_null<tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>*> \
          buffer,                                                         \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>&              \
          d_conformal_christoffel_second_kind,                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>& field_d_up,   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);    \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::spatial_ricci_tensor(                                             \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::iJkk<DTYPE(data), DIM(data), FRAME(data)>&              \
          d_conformal_christoffel_second_kind,                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>& field_d_up,   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
