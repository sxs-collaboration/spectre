// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
void apply_matrix_in_first_dim(double* result, const double* const input,
                               const Matrix& matrix, const size_t size) {
  dgemm_<true>('N', 'N',
               matrix.rows(),            // rows of matrix and result
               size / matrix.columns(),  // columns of result and input
               matrix.columns(),         // columns of matrix and rows of input
               1.0,                      // overall multiplier
               matrix.data(),            // matrix
               matrix.spacing(),         // rows of matrix including padding
               input,                    // input
               matrix.columns(),         // rows of input
               0.0,                      // overwrite output with result
               result,                   // result
               matrix.rows());           // rows of result
}
}  // namespace

template <typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const gsl::not_null<gsl::span<double>*> buffer,
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  static_assert(
      Dim > 0 and Dim < 4,
      "logical_partial_derivative is only implemented for 1, 2, and 3d");
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(buffer->size() >= num_grid_points,
         "The buffer in logical_partial_derivative must be at least of size "
             << num_grid_points << " but is of size " << buffer->size());

  set_number_of_grid_points(logical_derivative_of_u,
                            mesh.number_of_grid_points());
  const Matrix empty_matrix{};
  std::array<std::reference_wrapper<const Matrix>, Dim> diff_matrices{
      make_array<Dim, std::reference_wrapper<const Matrix>>(empty_matrix)};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(diff_matrices, d) =
        std::cref(Spectral::differentiation_matrix(mesh.slice_through(d)));
  }

  // It would be possible to check if the memory is contiguous and then
  // differentiate all components at once. Note that the buffer in that case
  // would also need to be the size of all components.
  for (size_t storage_index = 0; storage_index < u.size(); ++storage_index) {
    const auto u_tensor_index = u.get_tensor_index(storage_index);
    const auto xi_deriv_tensor_index = prepend(u_tensor_index, 0_st);
    apply_matrix_in_first_dim(
        logical_derivative_of_u->get(xi_deriv_tensor_index).data(),
        u[storage_index].data(), diff_matrices[0], num_grid_points);
    for (size_t i = 1; i < Dim; ++i) {
      const auto deriv_tensor_index = prepend(u_tensor_index, i);
      DataVector& deriv_component =
          logical_derivative_of_u->get(deriv_tensor_index);
      size_t chunk_size = diff_matrices[0].get().rows() *
                          (i == 1 ? 1 : gsl::at(diff_matrices, 1).get().rows());
      raw_transpose(make_not_null(deriv_component.data()),
                    u[storage_index].data(), chunk_size,
                    num_grid_points / chunk_size);
      apply_matrix_in_first_dim(buffer->data(), deriv_component.data(),
                                gsl::at(diff_matrices, i), num_grid_points);
      chunk_size = i == 1
                       ? (Dim == 2 ? gsl::at(diff_matrices, 1).get().rows()
                                   : gsl::at(diff_matrices, 1).get().rows() *
                                         gsl::at(diff_matrices, 2).get().rows())
                       : gsl::at(diff_matrices, 2).get().rows();
      raw_transpose(make_not_null(deriv_component.data()), buffer->data(),
                    chunk_size, num_grid_points / chunk_size);
    }
  }
}

template <typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  std::vector<double> buffer(mesh.number_of_grid_points());
  gsl::span<double> buffer_view{buffer.data(), buffer.size()};
  logical_partial_derivative(logical_derivative_of_u,
                             make_not_null(&buffer_view), u, mesh);
}

template <typename SymmList, typename IndexList, size_t Dim>
auto logical_partial_derivative(
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      result{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&result), u, mesh);
  return result;
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame>*>
        du,
    const TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>& logical_partial_derivative_of_u,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  for (size_t storage_index = 0;
       storage_index < Tensor<DataVector, SymmList, IndexList>::size();
       ++storage_index) {
    const auto u_multi_index =
        Tensor<DataVector, SymmList,
               IndexList>::structure::get_canonical_tensor_index(storage_index);
    for (size_t i = 0; i < Dim; i++) {
      const auto du_multi_index = prepend(u_multi_index, i);
      du->get(du_multi_index) =
          inverse_jacobian.get(0, i) *
          logical_partial_derivative_of_u.get(prepend(u_multi_index, 0_st));
      for (size_t j = 1; j < Dim; j++) {
        du->get(du_multi_index) +=
            inverse_jacobian.get(j, i) *
            logical_partial_derivative_of_u.get(prepend(u_multi_index, j));
      }
    }
  }
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame>*>
        du,
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      logical_partial_derivative_of_u{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&logical_partial_derivative_of_u), u,
                             mesh);
  partial_derivative<SymmList, IndexList>(du, logical_partial_derivative_of_u,
                                          inverse_jacobian);
}

template <typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
auto partial_derivative(
    const Tensor<DataVector, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo,
        DerivativeFrame> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataVector, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>
      result{mesh.number_of_grid_points()};
  partial_derivative(make_not_null(&result), u, mesh, inverse_jacobian);
  return result;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define GET_TENSOR(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<                                                           \
          TensorMetafunctions::prepend_spatial_index<                          \
              GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>,  \
          GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >                   \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<double>*> buffer,                                \
      const GET_TENSOR(data) < DataVector, GET_DIM(data),                      \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template void logical_partial_derivative(                                    \
      gsl::not_null<                                                           \
          TensorMetafunctions::prepend_spatial_index<                          \
              GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>,  \
          GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >                   \
          logical_derivative_of_u,                                             \
      const GET_TENSOR(data) < DataVector, GET_DIM(data),                      \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>,          \
      GET_DIM(data), UpLo::Lo,                                                 \
      Frame::ElementLogical >                                                  \
          logical_partial_derivative(const GET_TENSOR(data) < DataVector,      \
                                     GET_DIM(data), GET_FRAME(data) > &u,      \
                                     const Mesh<GET_DIM(data)>& mesh);         \
  template void partial_derivative<GET_TENSOR(data) < DataVector,              \
                                   GET_DIM(data), GET_FRAME(data)>::symmetry,  \
      GET_TENSOR(                                                              \
          data)<DataVector, GET_DIM(data), GET_FRAME(data)>::index_list >      \
          (const gsl::not_null<TensorMetafunctions::prepend_spatial_index<     \
                                   GET_TENSOR(data) < DataVector,              \
                                   GET_DIM(data), GET_FRAME(data)>,            \
                               GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* >    \
               du,                                                             \
           const TensorMetafunctions::prepend_spatial_index<                   \
               GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>, \
           GET_DIM(data), UpLo::Lo,                                            \
           Frame::ElementLogical > &logical_partial_derivative_of_u,           \
           const InverseJacobian<DataVector, GET_DIM(data),                    \
                                 Frame::ElementLogical, GET_FRAME(data)>       \
               &inverse_jacobian);                                             \
  template void partial_derivative(                                            \
      const gsl::not_null<                                                     \
          TensorMetafunctions::prepend_spatial_index<                          \
              GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>,  \
          GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* > du,                     \
      const GET_TENSOR(data) < DataVector, GET_DIM(data),                      \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,                   \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>,          \
      GET_DIM(data), UpLo::Lo,                                                 \
      GET_FRAME(data) >                                                        \
          partial_derivative(                                                  \
              const GET_TENSOR(data) < DataVector, GET_DIM(data),              \
              GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,           \
              const InverseJacobian<DataVector, GET_DIM(data),                 \
                                    Frame::ElementLogical, GET_FRAME(data)>&   \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (tnsr::a, tnsr::A, tnsr::i, tnsr::I, tnsr::ab, tnsr::Ab,
                         tnsr::aB, tnsr::AB, tnsr::ij, tnsr::iJ, tnsr::Ij,
                         tnsr::IJ, tnsr::iA, tnsr::ia, tnsr::aa, tnsr::AA,
                         tnsr::ii, tnsr::II, tnsr::ijj, tnsr::Ijj, tnsr::iaa))

#undef INSTANTIATION

#define INSTANTIATION(r, data)                                               \
  template void logical_partial_derivative(                                  \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<              \
          Scalar<DataVector>, GET_DIM(data), UpLo::Lo,                       \
          Frame::ElementLogical>*>                                           \
          logical_derivative_of_u,                                           \
      gsl::not_null<gsl::span<double>*> buffer, const Scalar<DataVector>& u, \
      const Mesh<GET_DIM(data)>& mesh);                                      \
  template void logical_partial_derivative(                                  \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<              \
          Scalar<DataVector>, GET_DIM(data), UpLo::Lo,                       \
          Frame::ElementLogical>*>                                           \
          logical_derivative_of_u,                                           \
      const Scalar<DataVector>& u, const Mesh<GET_DIM(data)>& mesh);         \
  template TensorMetafunctions::prepend_spatial_index<                       \
      Scalar<DataVector>, GET_DIM(data), UpLo::Lo, Frame::ElementLogical>    \
  logical_partial_derivative(const Scalar<DataVector>& u,                    \
                             const Mesh<GET_DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATE_JACOBIANS(r, data)                                        \
  template TensorMetafunctions::prepend_spatial_index<                        \
      GET_TENSOR(data) < DataVector, GET_DIM(data), Frame::ElementLogical,    \
      GET_FRAME(data)>,                                                       \
      GET_DIM(data), UpLo::Lo,                                                \
      GET_FRAME(data) >                                                       \
          partial_derivative(                                                 \
              const GET_TENSOR(data) < DataVector, GET_DIM(data),             \
              Frame::ElementLogical, GET_FRAME(data) > &u,                    \
              const Mesh<GET_DIM(data)>& mesh,                                \
              const InverseJacobian<DataVector, GET_DIM(data),                \
                                    Frame::ElementLogical, GET_FRAME(data)>&  \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIANS, (1, 2, 3), (Frame::Inertial),
                        (InverseJacobian))

#undef INSTANTIATE_JACOBIANS

#define INSTANTIATE_SCALAR(r, data)                                           \
  template void partial_derivative<Symmetry<>, index_list<>>(                 \
      gsl::not_null<tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>*> du, \
      const tnsr::i<DataVector, GET_DIM(data), Frame::ElementLogical>&        \
          logical_partial_derivative_of_u,                                    \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical, \
                            GET_FRAME(data)>& inverse_jacobian);              \
  template void partial_derivative(                                           \
      gsl::not_null<tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>*> du, \
      const Scalar<DataVector>& u, const Mesh<GET_DIM(data)>& mesh,           \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical, \
                            GET_FRAME(data)>& inverse_jacobian);              \
  template tnsr::i<DataVector, GET_DIM(data), GET_FRAME(data)>                \
  partial_derivative(                                                         \
      const Scalar<DataVector>& u, const Mesh<GET_DIM(data)>& mesh,           \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical, \
                            GET_FRAME(data)>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR, (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef INSTANTIATE_SCALAR
#undef GET_FRAME
#undef GET_DIM
#undef GET_TENSOR
