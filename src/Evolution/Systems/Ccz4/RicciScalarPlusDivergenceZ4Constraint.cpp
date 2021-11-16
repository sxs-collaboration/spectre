// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/RicciScalarPlusDivergenceZ4Constraint.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim, typename Frame, typename DataType>
void ricci_scalar_plus_divergence_z4_constraint(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& spatial_ricci_tensor,
    const tnsr::ij<DataType, Dim, Frame>& grad_spatial_z4_constraint) {
  destructive_resize_components(result,
                                get_size(get(conformal_factor_squared)));

  ::TensorExpressions::evaluate(
      result, conformal_factor_squared() *
                  inverse_conformal_spatial_metric(ti_I, ti_J) *
                  (spatial_ricci_tensor(ti_i, ti_j) +
                   grad_spatial_z4_constraint(ti_i, ti_j) +
                   grad_spatial_z4_constraint(ti_j, ti_i)));
}

template <size_t Dim, typename Frame, typename DataType>
Scalar<DataType> ricci_scalar_plus_divergence_z4_constraint(
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& spatial_ricci_tensor,
    const tnsr::ij<DataType, Dim, Frame>& grad_spatial_z4_constraint) {
  Scalar<DataType> result{};
  ricci_scalar_plus_divergence_z4_constraint(
      make_not_null(&result), conformal_factor_squared,
      inverse_conformal_spatial_metric, spatial_ricci_tensor,
      grad_spatial_z4_constraint);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                      \
  template void Ccz4::ricci_scalar_plus_divergence_z4_constraint( \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,           \
      const Scalar<DTYPE(data)>& conformal_factor_squared,        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&        \
          inverse_conformal_spatial_metric,                       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&        \
          spatial_ricci_tensor,                                   \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&        \
          grad_spatial_z4_constraint);                            \
  template Scalar<DTYPE(data)>                                    \
  Ccz4::ricci_scalar_plus_divergence_z4_constraint(               \
      const Scalar<DTYPE(data)>& conformal_factor_squared,        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&        \
          inverse_conformal_spatial_metric,                       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&        \
          spatial_ricci_tensor,                                   \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&        \
          grad_spatial_z4_constraint);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef FRAME
#undef DIM
