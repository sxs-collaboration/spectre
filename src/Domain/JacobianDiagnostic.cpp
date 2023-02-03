// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/JacobianDiagnostic.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
template <size_t Dim, typename Fr>
void jacobian_diagnostic(
    const gsl::not_null<
        tnsr::i<DataVector, Dim, typename Frame::ElementLogical>*>
        jacobian_diag,
    const Jacobian<DataVector, Dim, typename Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const TensorMetafunctions::prepend_spatial_index<
        tnsr::I<DataVector, Dim, Fr>, Dim, UpLo::Lo,
        typename Frame::ElementLogical>& numeric_jacobian_transpose) {
  const size_t number_of_points = analytic_jacobian.begin()->size();
  destructive_resize_components(jacobian_diag, number_of_points);

  // i_hat = logical frame index
  // i = mapped frame index
  Scalar<DataVector> jacobian_component_sum{number_of_points};
  for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {
    jacobian_diag->get(i_hat) = -abs(analytic_jacobian.get(0, i_hat));
    for (size_t i = 1; i < Dim; ++i) {
      jacobian_diag->get(i_hat) -= abs(analytic_jacobian.get(i, i_hat));
    }
    get(jacobian_component_sum) = abs(numeric_jacobian_transpose.get(i_hat, 0));
    for (size_t i = 1; i < Dim; ++i) {
      get(jacobian_component_sum) +=
          abs(numeric_jacobian_transpose.get(i_hat, i));
    }
    jacobian_diag->get(i_hat) /= get(jacobian_component_sum);
    jacobian_diag->get(i_hat) += 1.0;
  }
}

template <size_t Dim, typename Fr>
void jacobian_diagnostic(
    const gsl::not_null<
        tnsr::i<DataVector, Dim, typename Frame::ElementLogical>*>
        jacobian_diag,
    const ::Jacobian<DataVector, Dim, Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const tnsr::I<DataVector, Dim, Fr>& mapped_coords,
    const ::Mesh<Dim>& mesh) {
  // Note: Jacobian has the source frame index second, but
  // logical_partial_derivative prepends the logical (source frame) index.
  // So this is actually the transpose of the numerical jacobian.
  const auto numerical_jacobian_transpose =
      logical_partial_derivative(mapped_coords, mesh);

  jacobian_diagnostic(jacobian_diag, analytic_jacobian,
                      numerical_jacobian_transpose);
}

template <size_t Dim, typename Fr>
tnsr::i<DataVector, Dim, typename Frame::ElementLogical> jacobian_diagnostic(
    const ::Jacobian<DataVector, Dim, Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const tnsr::I<DataVector, Dim, Fr>& mapped_coords,
    const ::Mesh<Dim>& mesh) {
  tnsr::i<DataVector, Dim, typename Frame::ElementLogical> jacobian_diag{};
  jacobian_diagnostic(make_not_null(&jacobian_diag), analytic_jacobian,
                      mapped_coords, mesh);
  return jacobian_diag;
}
}  // namespace domain

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void domain::jacobian_diagnostic(                                \
      const gsl::not_null<                                                  \
          tnsr::i<DataVector, DIM(data), typename Frame::ElementLogical>*>  \
          jacobian_diag,                                                    \
      const Jacobian<DataVector, DIM(data), Frame::ElementLogical,          \
                     FRAME(data)>& analytic_jacobian,                       \
      const TensorMetafunctions::prepend_spatial_index<                     \
          tnsr::I<DataVector, DIM(data), FRAME(data)>, DIM(data), UpLo::Lo, \
          typename Frame::ElementLogical>& numeric_jacobian_transpose);     \
  template void domain::jacobian_diagnostic(                                \
      const gsl::not_null<                                                  \
          tnsr::i<DataVector, DIM(data), typename Frame::ElementLogical>*>  \
          jacobian_diag,                                                    \
      const ::Jacobian<DataVector, DIM(data), Frame::ElementLogical,        \
                       FRAME(data)>& analytic_jacobian,                     \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& mapped_coords,     \
      const ::Mesh<DIM(data)>& mesh);                                       \
  template tnsr::i<DataVector, DIM(data), typename Frame::ElementLogical>   \
  domain::jacobian_diagnostic(                                              \
      const ::Jacobian<DataVector, DIM(data), Frame::ElementLogical,        \
                       FRAME(data)>& analytic_jacobian,                     \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& mapped_coords,     \
      const ::Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
