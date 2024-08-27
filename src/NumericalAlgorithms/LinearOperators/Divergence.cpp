// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename DataType, size_t Dim, typename DerivativeFrame>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, DerivativeFrame>;
};
}  // namespace

template <typename DataType, size_t Dim, typename DerivativeFrame>
Scalar<DataType> divergence(
    const tnsr::I<DataType, Dim, DerivativeFrame>& input, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  Scalar<DataType> div_input{mesh.number_of_grid_points()};
  divergence(make_not_null(&div_input), input, mesh, inverse_jacobian);
  return div_input;
}

template <typename DataType, size_t Dim, typename DerivativeFrame>
void divergence(const gsl::not_null<Scalar<DataType>*> div_input,
                const tnsr::I<DataType, Dim, DerivativeFrame>& input,
                const Mesh<Dim>& mesh,
                const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                      DerivativeFrame>& inverse_jacobian) {
  set_number_of_grid_points(div_input, mesh.number_of_grid_points());

  // We have to copy into a Variables because we don't currently have partial
  // derivative functions for anything other than Variables.
  using VectorTag = VectorTag<DataType, Dim, DerivativeFrame>;
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

template <typename ResultTensor, typename FluxTensor, size_t Dim>
void logical_divergence(const gsl::not_null<ResultTensor*> div_flux,
                        const FluxTensor& flux, const Mesh<Dim>& mesh) {
  // Note: This function hasn't been optimized much at all. Feel free to
  // optimize if needed!
  static const Matrix identity_matrix{};
  for (size_t d = 0; d < Dim; ++d) {
    auto matrices = make_array<Dim>(std::cref(identity_matrix));
    gsl::at(matrices, d) =
        Spectral::differentiation_matrix(mesh.slice_through(d));
    for (size_t storage_index = 0; storage_index < div_flux->size();
         ++storage_index) {
      const auto div_flux_index = div_flux->get_tensor_index(storage_index);
      const auto flux_index = prepend(div_flux_index, d);
      div_flux->get(div_flux_index) +=
          apply_matrices(matrices, flux.get(flux_index), mesh.extents());
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void divergence(                                               \
      const gsl::not_null<Scalar<DTYPE(data)>*> div_input,                \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& input,          \
      const Mesh<DIM(data)>& mesh,                                        \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian);              \
  template Scalar<DTYPE(data)> divergence(                                \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& input,          \
      const Mesh<DIM(data)>& mesh,                                        \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3),
                        (Frame::Grid, Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TENSOR(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION_SCALAR(r, data)                                     \
  template void logical_divergence(                                       \
      const gsl::not_null<Scalar<DTYPE(data)>*> div_flux,                 \
      const tnsr::I<DTYPE(data), DIM(data), Frame::ElementLogical>& flux, \
      const Mesh<DIM(data)>& mesh);
#define INSTANTIATION_TENSOR(r, data)                                    \
  template void logical_divergence(                                      \
      const gsl::not_null<tnsr::TENSOR(data) < DTYPE(data), DIM(data),   \
                          Frame::Inertial>* > div_flux,                  \
      const TensorMetafunctions::prepend_spatial_index<                  \
          tnsr::TENSOR(data) < DTYPE(data), DIM(data), Frame::Inertial>, \
      DIM(data), UpLo::Up, Frame::ElementLogical > &flux,                \
      const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION_SCALAR, (DataVector, ComplexDataVector),
                        (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATION_TENSOR, (DataVector, ComplexDataVector),
                        (1, 2, 3), (i, I))

#undef INSTANTIATION
