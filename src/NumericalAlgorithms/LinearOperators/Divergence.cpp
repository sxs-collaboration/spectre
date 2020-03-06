// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename DerivativeFrame>
struct VectorTag {
  using type = tnsr::I<DataVector, Dim, DerivativeFrame>;
};
}  // namespace

template <size_t Dim, typename DerivativeFrame>
Scalar<DataVector> divergence(
    const tnsr::I<DataVector, Dim, DerivativeFrame>& input,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept {
  // We have to copy into a Variables because we don't currently have partial
  // derivative functions for anything other than Variables.
  using VectorTag = VectorTag<Dim, DerivativeFrame>;
  Variables<tmpl::list<VectorTag>> vars{get<0>(input).size()};
  get<VectorTag>(vars) = input;
  const auto logical_derivs =
      logical_partial_derivatives<tmpl::list<VectorTag>>(vars, mesh);

  Scalar<DataVector> div_input{get<0>(input).size(), 0.0};
  for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
    for (size_t deriv_i = 0; deriv_i < Dim; ++deriv_i) {
      get(div_input) +=
          inverse_jacobian.get(logical_i, deriv_i) *
          get<VectorTag>(gsl::at(logical_derivs, logical_i)).get(deriv_i);
    }
  }

  return div_input;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                       \
  template Scalar<DataVector> divergence(                          \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& input,    \
      const Mesh<DIM(data)>& mesh,                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical, \
                            FRAME(data)>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
