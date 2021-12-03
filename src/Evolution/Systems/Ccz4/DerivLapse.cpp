// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/DerivLapse.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void grad_grad_lapse(
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> result,
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) {
  destructive_resize_components(result, get_size(get(lapse)));

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      result->get(i, j) = field_a.get(i) * field_a.get(j) +
                          0.5 * (d_field_a.get(i, j) + d_field_a.get(j, i));
      for (size_t k = 0; k < Dim; ++k) {
        result->get(i, j) -=
            christoffel_second_kind.get(k, i, j) * field_a.get(k);
      }
      result->get(i, j) *= get(lapse);
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> grad_grad_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) {
  tnsr::ij<DataType, Dim, Frame> result{};
  grad_grad_lapse(make_not_null(&result), lapse, christoffel_second_kind,
                  field_a, d_field_a);
  return result;
}

template <size_t Dim, typename Frame, typename DataType>
void divergence_lapse(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_metric,
    const tnsr::ij<DataType, Dim, Frame>& grad_grad_lapse) {
  destructive_resize_components(result,
                                get_size(get(conformal_factor_squared)));

  get(*result) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      get(*result) +=
          inverse_conformal_metric.get(i, j) * grad_grad_lapse.get(i, j);
    }
  }
  get(*result) *= get(conformal_factor_squared);
}

template <size_t Dim, typename Frame, typename DataType>
Scalar<DataType> divergence_lapse(
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_metric,
    const tnsr::ij<DataType, Dim, Frame>& grad_grad_lapse) {
  Scalar<DataType> result{};
  divergence_lapse(make_not_null(&result), conformal_factor_squared,
                   inverse_conformal_metric, grad_grad_lapse);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template void Ccz4::grad_grad_lapse(                                       \
      const gsl::not_null<tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>*>    \
          result,                                                            \
      const Scalar<DTYPE(data)>& lapse,                                      \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                  \
          christoffel_second_kind,                                           \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_a,           \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_a);       \
  template tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>                     \
  Ccz4::grad_grad_lapse(                                                     \
      const Scalar<DTYPE(data)>& lapse,                                      \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                  \
          christoffel_second_kind,                                           \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_a,           \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& d_field_a);       \
  template void Ccz4::divergence_lapse(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                      \
      const Scalar<DTYPE(data)>& conformal_factor_squared,                   \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_conformal_metric,                                          \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& grad_grad_lapse); \
  template Scalar<DTYPE(data)> Ccz4::divergence_lapse(                       \
      const Scalar<DTYPE(data)>& conformal_factor_squared,                   \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_conformal_metric,                                          \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>& grad_grad_lapse);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
