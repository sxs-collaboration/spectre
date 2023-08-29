// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename DerivativeFrame>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, DerivativeFrame>;
};
}  // namespace

template <size_t Dim, typename DerivativeFrame>
Scalar<DataVector> divergence(
    const tnsr::I<DataVector, Dim, DerivativeFrame>& input,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  Scalar<DataVector> div_input{mesh.number_of_grid_points()};
  divergence(make_not_null(&div_input), input, mesh, inverse_jacobian);
  return div_input;
}

template <size_t Dim, typename DerivativeFrame>
void divergence(
    const gsl::not_null<Scalar<DataVector>*> div_input,
    const tnsr::I<DataVector, Dim, DerivativeFrame>& input,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  set_number_of_grid_points(div_input, mesh.number_of_grid_points());

  // We have to copy into a Variables because we don't currently have partial
  // derivative functions for anything other than Variables.
  using VectorTag = VectorTag<Dim, DerivativeFrame>;
  Variables<tmpl::list<VectorTag>> vars{get<0>(input).size()};
  get<VectorTag>(vars) = input;
  const auto logical_derivs =
      logical_partial_derivatives<tmpl::list<VectorTag>>(vars, mesh);

  get(*div_input) = 0.0;
  for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
    for (size_t deriv_i = 0; deriv_i < Dim; ++deriv_i) {
      get(*div_input) +=
          inverse_jacobian.get(logical_i, deriv_i) *
          get<VectorTag>(gsl::at(logical_derivs, logical_i)).get(deriv_i);
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template void divergence(                                               \
    const gsl::not_null<Scalar<DataVector>*> div_input,                   \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& input,           \
      const Mesh<DIM(data)>& mesh,                                        \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian);              \
  template Scalar<DataVector> divergence(                                 \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& input,           \
      const Mesh<DIM(data)>& mesh,                                        \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
