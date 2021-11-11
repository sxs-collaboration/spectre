// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/DerivZ4Constraint.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void grad_spatial_z4_constraint(
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> result,
    const tnsr::i<DataType, Dim, Frame>& spatial_z4_constraint,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::I<DataType, Dim, Frame>&
        gamma_hat_minus_contracted_conformal_christoffel,
    const tnsr::iJ<DataType, Dim, Frame>&
        d_gamma_hat_minus_contracted_conformal_christoffel) {
  destructive_resize_components(result,
                                get_size(get<0, 0>(conformal_spatial_metric)));

  ::TensorExpressions::evaluate<ti_i, ti_j>(
      result,
      field_d(ti_i, ti_j, ti_l) *
              gamma_hat_minus_contracted_conformal_christoffel(ti_L) +
          0.5 * conformal_spatial_metric(ti_j, ti_l) *
              d_gamma_hat_minus_contracted_conformal_christoffel(ti_i, ti_L) -
          christoffel_second_kind(ti_L, ti_i, ti_j) *
              spatial_z4_constraint(ti_l));
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> grad_spatial_z4_constraint(
    const tnsr::i<DataType, Dim, Frame>& spatial_z4_constraint,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::I<DataType, Dim, Frame>&
        gamma_hat_minus_contracted_conformal_christoffel,
    const tnsr::iJ<DataType, Dim, Frame>&
        d_gamma_hat_minus_contracted_conformal_christoffel) {
  tnsr::ij<DataType, Dim, Frame> result{};
  grad_spatial_z4_constraint(
      make_not_null(&result), spatial_z4_constraint, conformal_spatial_metric,
      christoffel_second_kind, field_d,
      gamma_hat_minus_contracted_conformal_christoffel,
      d_gamma_hat_minus_contracted_conformal_christoffel);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void Ccz4::grad_spatial_z4_constraint(                         \
      const gsl::not_null<tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                         \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                 \
          spatial_z4_constraint,                                          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                 \
          gamma_hat_minus_contracted_conformal_christoffel,               \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_gamma_hat_minus_contracted_conformal_christoffel);            \
  template tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::grad_spatial_z4_constraint(                                       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                 \
          spatial_z4_constraint,                                          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          christoffel_second_kind,                                        \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& field_d,      \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                 \
          gamma_hat_minus_contracted_conformal_christoffel,               \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_gamma_hat_minus_contracted_conformal_christoffel);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
