// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Ccz4/Ricci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <typename DataType, size_t Dim, typename Frame>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& contracted_christoffel_second_kind,
    const tnsr::ij<DataType, Dim, Frame>&
        contracted_d_conformal_christoffel_difference,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::I<DataType, Dim, Frame>& contracted_field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p) {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));

  tenex::evaluate<ti::i, ti::j>(
      result,
      contracted_d_conformal_christoffel_difference(ti::i, ti::j) +
          // Add terms of \partial_m \Gamma^m_{ij} and
          // -\partial_j \Gamma^m_{im} that have a coefficient of 2
          2.0 * (contracted_field_d_up(ti::L) *
                     (conformal_spatial_metric(ti::j, ti::l) * field_p(ti::i) +
                      conformal_spatial_metric(ti::i, ti::l) * field_p(ti::j) -
                      conformal_spatial_metric(ti::i, ti::j) * field_p(ti::l)) -
                 inverse_conformal_spatial_metric(ti::M, ti::L) *
                     (field_d(ti::m, ti::j, ti::l) * field_p(ti::i) +
                      field_d(ti::m, ti::i, ti::l) * field_p(ti::j) -
                      field_d(ti::m, ti::i, ti::j) * field_p(ti::l)) -
                 field_d_up(ti::j, ti::M, ti::L) *
                     (conformal_spatial_metric(ti::m, ti::l) * field_p(ti::i) +
                      conformal_spatial_metric(ti::i, ti::l) * field_p(ti::m) -
                      conformal_spatial_metric(ti::i, ti::m) * field_p(ti::l)) +
                 inverse_conformal_spatial_metric(ti::M, ti::L) *
                     (field_d(ti::j, ti::m, ti::l) * field_p(ti::i) +
                      field_d(ti::j, ti::i, ti::l) * field_p(ti::m) -
                      field_d(ti::j, ti::i, ti::m) * field_p(ti::l))) -
          // Add \partial_{(i} P_{j)} type terms
          0.5 * inverse_conformal_spatial_metric(ti::M, ti::L) *
              (conformal_spatial_metric(ti::j, ti::l) *
                   (d_field_p(ti::m, ti::i) + d_field_p(ti::i, ti::m)) +
               conformal_spatial_metric(ti::i, ti::l) *
                   (d_field_p(ti::m, ti::j) + d_field_p(ti::j, ti::m)) -
               conformal_spatial_metric(ti::i, ti::j) *
                   (d_field_p(ti::m, ti::l) + d_field_p(ti::l, ti::m)) -
               conformal_spatial_metric(ti::m, ti::l) *
                   (d_field_p(ti::j, ti::i) + d_field_p(ti::i, ti::j)) -
               conformal_spatial_metric(ti::i, ti::l) *
                   (d_field_p(ti::j, ti::m) + d_field_p(ti::m, ti::j)) +
               conformal_spatial_metric(ti::i, ti::m) *
                   (d_field_p(ti::j, ti::l) + d_field_p(ti::l, ti::j))) +
          // Add last two terms for R_{ij}
          christoffel_second_kind(ti::L, ti::i, ti::j) *
              contracted_christoffel_second_kind(ti::l) -
          christoffel_second_kind(ti::L, ti::i, ti::m) *
              christoffel_second_kind(ti::M, ti::l, ti::j));
}

template <typename DataType, size_t Dim, typename Frame>
tnsr::ii<DataType, Dim, Frame> spatial_ricci_tensor(
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& contracted_christoffel_second_kind,
    const tnsr::ij<DataType, Dim, Frame>&
        contracted_d_conformal_christoffel_difference,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::I<DataType, Dim, Frame>& contracted_field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p) {
  tnsr::ii<DataType, Dim, Frame> result{};
  spatial_ricci_tensor(make_not_null(&result), christoffel_second_kind,
                       contracted_christoffel_second_kind,
                       contracted_d_conformal_christoffel_difference,
                       conformal_spatial_metric,
                       inverse_conformal_spatial_metric, field_d, field_d_up,
                       contracted_field_d_up, field_p, d_field_p);
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
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                 \
          contracted_christoffel_second_kind,                             \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          contracted_d_conformal_christoffel_difference,                  \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>& field_d_up,   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                 \
          contracted_field_d_up,                                          \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);    \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::spatial_ricci_tensor(                                             \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                 \
          contracted_christoffel_second_kind,                             \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          contracted_d_conformal_christoffel_difference,                  \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>& field_d_up,   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                 \
          contracted_field_d_up,                                          \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);

// Instantiations are split into several compilation units to reduce
// compiler memory consumption.
