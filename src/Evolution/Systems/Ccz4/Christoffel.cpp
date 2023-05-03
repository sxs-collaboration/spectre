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
template <typename DataType, size_t Dim, typename Frame>
void conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d) {
  destructive_resize_components(
      result, get_size(get<0, 0>(inverse_conformal_spatial_metric)));

  ::tenex::evaluate<ti::K, ti::i, ti::j>(
      result, inverse_conformal_spatial_metric(ti::K, ti::L) *
                  (field_d(ti::i, ti::j, ti::l) + field_d(ti::j, ti::i, ti::l) -
                   field_d(ti::l, ti::i, ti::j)));
}

template <typename DataType, size_t Dim, typename Frame>
tnsr::Ijj<DataType, Dim, Frame> conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d) {
  tnsr::Ijj<DataType, Dim, Frame> result{};
  conformal_christoffel_second_kind(make_not_null(&result),
                                    inverse_conformal_spatial_metric, field_d);
  return result;
}

template <typename DataType, size_t Dim, typename Frame>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, Dim, Frame>*> result,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind) {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));

  ::tenex::evaluate<ti::K, ti::i, ti::j>(
      result,
      conformal_christoffel_second_kind(ti::K, ti::i, ti::j) -
          inverse_conformal_spatial_metric(ti::K, ti::L) *
              (conformal_spatial_metric(ti::j, ti::l) * field_p(ti::i) +
               conformal_spatial_metric(ti::i, ti::l) * field_p(ti::j) -
               conformal_spatial_metric(ti::i, ti::j) * field_p(ti::l)));
}

template <typename DataType, size_t Dim, typename Frame>
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

template <typename DataType, size_t Dim, typename Frame>
void contracted_conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind) {
  destructive_resize_components(
      result, get_size(get<0, 0>(inverse_conformal_spatial_metric)));

  ::tenex::evaluate<ti::I>(
      result, inverse_conformal_spatial_metric(ti::J, ti::L) *
                  conformal_christoffel_second_kind(ti::I, ti::j, ti::l));
}

template <typename DataType, size_t Dim, typename Frame>
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
