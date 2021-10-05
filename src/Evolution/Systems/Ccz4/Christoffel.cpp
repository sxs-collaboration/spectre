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
    const tnsr::ijj<DataType, Dim, Frame>& field_d) {
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
    const tnsr::ijj<DataType, Dim, Frame>& field_d) {
  tnsr::Ijj<DataType, Dim, Frame> result{};
  conformal_christoffel_second_kind(make_not_null(&result),
                                    inverse_conformal_spatial_metric, field_d);
  return result;
}

template <size_t Dim, typename Frame, typename DataType>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, Dim, Frame>*> result,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind) {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));

  ::TensorExpressions::evaluate<ti_K, ti_i, ti_j>(
      result, conformal_christoffel_second_kind(ti_K, ti_i, ti_j) -
                  inverse_conformal_spatial_metric(ti_K, ti_L) *
                      (conformal_spatial_metric(ti_j, ti_l) * field_p(ti_i) +
                       conformal_spatial_metric(ti_i, ti_l) * field_p(ti_j) -
                       conformal_spatial_metric(ti_i, ti_j) * field_p(ti_l)));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::Ijj<DataType, Dim, Frame> christoffel_second_kind(
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind) {
  tnsr::Ijj<DataType, Dim, Frame> result{};
  christoffel_second_kind(make_not_null(&result), conformal_spatial_metric,
                          inverse_conformal_spatial_metric, field_p,
                          conformal_christoffel_second_kind);
  return result;
}

template <size_t Dim, typename Frame, typename DataType>
void contracted_conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind) {
  destructive_resize_components(
      result, get_size(get<0, 0>(inverse_conformal_spatial_metric)));

  ::TensorExpressions::evaluate<ti_I>(
      result, inverse_conformal_spatial_metric(ti_J, ti_L) *
                  conformal_christoffel_second_kind(ti_I, ti_j, ti_l));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::I<DataType, Dim, Frame> contracted_conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind) {
  tnsr::I<DataType, Dim, Frame> result{};
  contracted_conformal_christoffel_second_kind(
      make_not_null(&result), inverse_conformal_spatial_metric,
      conformal_christoffel_second_kind);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void Ccz4::conformal_christoffel_second_kind(                   \
      const gsl::not_null<tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_conformal_spatial_metric,                                \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d);      \
  template tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::conformal_christoffel_second_kind(                                 \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_conformal_spatial_metric,                                \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d);      \
  template void Ccz4::christoffel_second_kind(                             \
      const gsl::not_null<tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          conformal_spatial_metric,                                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_conformal_spatial_metric,                                \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,         \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_christoffel_second_kind);                              \
  template tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::christoffel_second_kind(                                           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          conformal_spatial_metric,                                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_conformal_spatial_metric,                                \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,         \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_christoffel_second_kind);                              \
  template void Ccz4::contracted_conformal_christoffel_second_kind(        \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>   \
          result,                                                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_conformal_spatial_metric,                                \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_christoffel_second_kind);                              \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)>                    \
  Ccz4::contracted_conformal_christoffel_second_kind(                      \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_conformal_spatial_metric,                                \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_christoffel_second_kind);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
