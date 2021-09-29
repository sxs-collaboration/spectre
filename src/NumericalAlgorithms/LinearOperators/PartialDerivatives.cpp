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
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void apply_matrix_in_first_dim(double* result, const double* const input,
                               const Matrix& matrix,
                               const size_t size) noexcept {
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
    const Tensor<DataVector, SymmList, IndexList>& u,
    const Mesh<Dim>& mesh) noexcept {
  static_assert(
      Dim > 0 and Dim < 4,
      "logical_partial_derivative is only implemented for 1, 2, and 3d");
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(buffer->size() >= num_grid_points,
         "The buffer in logical_partial_derivative must be at least of size "
             << num_grid_points << " but is of size " << buffer->size());

  destructive_resize_components(logical_derivative_of_u,
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
    const Tensor<DataVector, SymmList, IndexList>& u,
    const Mesh<Dim>& mesh) noexcept {
  std::vector<double> buffer(mesh.number_of_grid_points());
  gsl::span<double> buffer_view{buffer.data(), buffer.size()};
  logical_partial_derivative(logical_derivative_of_u,
                             make_not_null(&buffer_view), u, mesh);
}

template <typename SymmList, typename IndexList, size_t Dim>
auto logical_partial_derivative(
    const Tensor<DataVector, SymmList, IndexList>& u,
    const Mesh<Dim>& mesh) noexcept
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

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_TENSOR(data) BOOST_PP_TUPLE_ELEM(1, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                                \
  template void logical_partial_derivative(                                   \
      gsl::not_null<                                                          \
          TensorMetafunctions::prepend_spatial_index<                         \
              GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>, \
          GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >                  \
          logical_derivative_of_u,                                            \
      gsl::not_null<gsl::span<double>*> buffer,                               \
      const GET_TENSOR(data) < DataVector, GET_DIM(data),                     \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh) noexcept;        \
  template void logical_partial_derivative(                                   \
      gsl::not_null<                                                          \
          TensorMetafunctions::prepend_spatial_index<                         \
              GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>, \
          GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >                  \
          logical_derivative_of_u,                                            \
      const GET_TENSOR(data) < DataVector, GET_DIM(data),                     \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh) noexcept;        \
  template TensorMetafunctions::prepend_spatial_index<                        \
      GET_TENSOR(data) < DataVector, GET_DIM(data), GET_FRAME(data)>,         \
      GET_DIM(data), UpLo::Lo,                                                \
      Frame::ElementLogical > logical_partial_derivative(                     \
                                  const GET_TENSOR(data) < DataVector,        \
                                  GET_DIM(data), GET_FRAME(data) > &u,        \
                                  const Mesh<GET_DIM(data)>& mesh) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (tnsr::a, tnsr::A, tnsr::i, tnsr::I, tnsr::ab, tnsr::Ab,
                         tnsr::aB, tnsr::AB, tnsr::ij, tnsr::iJ, tnsr::Ij,
                         tnsr::IJ, tnsr::iA, tnsr::ia, tnsr::aa, tnsr::AA,
                         tnsr::ii, tnsr::II),
                        (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef GET_FRAME
#undef GET_TENSOR

#define INSTANTIATION(r, data)                                                \
  template void logical_partial_derivative(                                   \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<               \
          Scalar<DataVector>, GET_DIM(data), UpLo::Lo,                        \
          Frame::ElementLogical>*>                                            \
          logical_derivative_of_u,                                            \
      gsl::not_null<gsl::span<double>*> buffer, const Scalar<DataVector>& u,  \
      const Mesh<GET_DIM(data)>& mesh) noexcept;                              \
  template void logical_partial_derivative(                                   \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<               \
          Scalar<DataVector>, GET_DIM(data), UpLo::Lo,                        \
          Frame::ElementLogical>*>                                            \
          logical_derivative_of_u,                                            \
      const Scalar<DataVector>& u, const Mesh<GET_DIM(data)>& mesh) noexcept; \
  template TensorMetafunctions::prepend_spatial_index<                        \
      Scalar<DataVector>, GET_DIM(data), UpLo::Lo, Frame::ElementLogical>     \
  logical_partial_derivative(const Scalar<DataVector>& u,                     \
                             const Mesh<GET_DIM(data)>& mesh) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef GET_DIM
