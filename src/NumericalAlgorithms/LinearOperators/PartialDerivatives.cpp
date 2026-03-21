// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include <array>
#include <blaze/math/DynamicMatrix.h>
#include <cstddef>
#include <functional>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace partial_derivatives_detail {
void apply_matrix_in_first_dim(double* result, const double* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  dgemm_<true>(
      'N', 'N',
      matrix.rows(),              // rows of matrix and result
      size / matrix.columns(),    // columns of result and input
      matrix.columns(),           // columns of matrix and rows of input
      1.0,                        // overall multiplier
      matrix.data(),              // matrix
      matrix.spacing(),           // rows of matrix including padding
      input,                      // input
      matrix.columns(),           // rows of input
      add_to_result ? 1.0 : 0.0,  // overwrite output with result or add to it
      result,                     // result
      matrix.rows());             // rows of result
}
void apply_matrix_in_first_dim(std::complex<double>* result,
                               const std::complex<double>* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  // BLAS zgemm operates on complex matrices, so we need to copy the real matrix
  // to a complex matrix with zero imaginary part before calling zgemm.
  // Possible performance optimization: avoid the copy here by storing the
  // complex matrix in a static cache. We probably only want to add this to
  // Spectral.hpp once profiling shows that it becomes necessary.
  const blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>
      matrix_complex{matrix};
  zgemm_<true>('N', 'N',
               matrix.rows(),            // rows of matrix and result
               size / matrix.columns(),  // columns of result and input
               matrix.columns(),         // columns of matrix and rows of input
               std::complex{1.0, 0.0},   // overall multiplier
               matrix_complex.data(),    // matrix
               matrix.spacing(),         // rows of matrix including padding
               input,                    // input
               matrix.columns(),         // rows of input
               std::complex{add_to_result ? 1.0 : 0.0,
                            0.0},  // overwrite output with result or add to it
               result,             // result
               matrix.rows());     // rows of result
  // This implementation is ~1.35x slower than the implementation above (based
  // on the "Partial derivatives complex" benchmark in
  // Test_PartialDerivatives.cpp).
  //   DataVector buffer(size * 2);
  //   raw_transpose(make_not_null(reinterpret_cast<double*>(result)),
  //                 reinterpret_cast<const double*>(input), 2, size);
  //   apply_matrix_in_first_dim(buffer.data(),
  //                             reinterpret_cast<const double*>(result),
  //                             matrix, size * 2, add_to_result);
  //   raw_transpose(make_not_null(reinterpret_cast<double*>(result)),
  //                 buffer.data(), size, 2);
}
}  // namespace partial_derivatives_detail

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const gsl::not_null<gsl::span<typename DataType::value_type>*> buffer,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  static_assert(
      Dim > 0 and Dim < 4,
      "logical_partial_derivative is only implemented for 1, 2, and 3d");
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(buffer->size() >= num_grid_points,
         "The buffer in logical_partial_derivative must be at least of size "
             << num_grid_points << " but is of size " << buffer->size());

  set_number_of_grid_points(logical_derivative_of_u,
                            mesh.number_of_grid_points());
  if (Dim == 3 and mesh.basis(1) == Spectral::Basis::SphericalHarmonic) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      const Matrix& differentiation_matrix_xi =
          Spectral::differentiation_matrix(mesh.slice_through(0));
      const auto& ylm = ylm::get_spherepack_cache(mesh.extents(1) - 1);
      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 0_st)).data(),
            u[storage_index].data(), differentiation_matrix_xi,
            num_grid_points);
        const auto du = std::array{
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 1_st)).data(),
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 2_st)).data()};
        ylm.gradient_all_offsets(du, make_not_null(u[storage_index].data()),
                                 mesh.extents(0));
      }
    } else {
      ERROR(
          "Support for complex numbers with spherical harmonics is not yet "
          "implemented for logical_partial_derivative.");
    }
  } else if (Dim == 2 and mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
                 mesh.basis(1) == Spectral::Basis::ZernikeB2,
             "Unexpected basis combination: " << mesh.basis());
      const size_t n_r = mesh.extents(0);
      const size_t n_phi = mesh.extents(1);
      const size_t n_r_max = 2 * n_r - 2;
      ASSERT(
          n_phi % 2 == 1,
          "Fourier with an even number of grid points can be unstable due to "
          "the top derivative not being representable");
      ASSERT(n_phi / 2 <= n_r_max,
             "Zernike & Fourier on a disk have angular resolution limited by "
             "extents in both dimensions. We choose to enforce the restriction "
             "that the Fourier modal space is not larger than the Zernike "
             "angular capabilities (which would waste space and be physically "
             "ill-motivated).\nn_phi / 2 = "
                 << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);
      const size_t M = n_phi / 2;

      const Matrix& diff_matrix_r_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& diff_matrix_r_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);
      const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto r_deriv_tensor_index = prepend(u_tensor_index, 0_st);
        const auto phi_deriv_tensor_index = prepend(u_tensor_index, 1_st);

        // Do phi derivative
        dgemm_<true>(
            'N',
            'T',  // transpose differentiation matrix to act on rows of data
            n_r, n_phi, n_phi, 1.0, u[storage_index].data(), n_r,
            diff_matrix_phi.data(), diff_matrix_phi.spacing(), 0.0,
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(phi_deriv_tensor_index).data(), n_r);

        // Get u in terms of r and m index
        dgemm_<true>('N', 'T', n_r, n_phi, n_phi, 1.0, u[storage_index].data(),
                     n_r, nodal_to_modal_phi.data(),
                     nodal_to_modal_phi.spacing(), 0.0,
                     // NOLINTNEXTLINE(readability-redundant-smartptr-get)
                     logical_derivative_of_u->get(r_deriv_tensor_index).data(),
                     n_r);  // C and ldc

        // Do radial derivative
        // m = 0
        dgemv_('N', n_r, n_r, 1.0, diff_matrix_r_even.data(),
               diff_matrix_r_even.spacing(),
               // NOLINTNEXTLINE(readability-redundant-smartptr-get)
               logical_derivative_of_u->get(r_deriv_tensor_index).data(), 1,
               0.0,  // beta = 0.0
               (*buffer).data(), 1);
        // j is index into Fourier modal vector
        // {u_0, u_1, u_{-1}, u_2, u_{-2}, ... , u_M, u_{-M}}
        size_t j = 1;
        for (size_t m = 1; m <= M; ++m) {
          const Matrix& diff_r =
              m % 2 == 0 ? diff_matrix_r_even : diff_matrix_r_odd;
          dgemm_<true>(
              'N', 'N', n_r, 2, n_r, 1.0, diff_r.data(), diff_r.spacing(),
              // NOLINTNEXTLINE(readability-redundant-smartptr-get)
              logical_derivative_of_u->get(r_deriv_tensor_index).data() +
                  j * n_r,
              n_r,
              0.0,  // beta = 0.0
              // NOLINTNEXTLINE
              (*buffer).data() + j * n_r, n_r);
          j += 2;
        }
        // Transform from m back to phi
        dgemm_<true>(
            'N', 'T', n_r, n_phi, n_phi, 1.0, (*buffer).data(), n_r,
            modal_to_nodal_phi.data(), modal_to_nodal_phi.spacing(), 0.0,
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(r_deriv_tensor_index).data(), n_r);
      }
    } else {
      ERROR(
          "Support for complex numbers with disk is not yet implemented for "
          "logical_partial_derivative.");
    }
  } else if (Dim == 3 and mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
                 mesh.basis(1) == Spectral::Basis::ZernikeB2,
             "Unexpected basis combination: " << mesh.basis());
      const size_t n_r = mesh.extents(0);
      const size_t n_phi = mesh.extents(1);
      const size_t n_z = mesh.extents(2);
      const size_t n_r_max = 2 * n_r - 2;
      ASSERT(
          n_phi % 2 == 1,
          "Fourier with an even number of grid points can be unstable due to "
          "the top derivative not being representable");
      ASSERT(n_phi / 2 <= n_r_max,
             "Zernike & Fourier on a disk have angular resolution limited by "
             "extents in both dimensions. We choose to enforce the restriction "
             "that the Fourier modal space is not larger than the Zernike "
             "angular capabilities (which would waste space and be physically "
             "ill-motivated).\nn_phi / 2 = "
                 << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);

      const Matrix& diff_matrix_r_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& diff_matrix_r_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);
      const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& diff_matrix_z =
          Spectral::differentiation_matrix(mesh.slice_through(2));

      const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const DataType& u_component = u[storage_index];
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto r_deriv_tensor_index = prepend(u_tensor_index, 0_st);
        DataType& r_deriv_component =
            logical_derivative_of_u->get(r_deriv_tensor_index);
        const auto phi_deriv_tensor_index = prepend(u_tensor_index, 1_st);
        DataType& phi_deriv_component =
            logical_derivative_of_u->get(phi_deriv_tensor_index);
        const auto z_deriv_tensor_index = prepend(u_tensor_index, 2_st);
        DataType& z_deriv_component =
            logical_derivative_of_u->get(z_deriv_tensor_index);

        // phi derivative
        raw_transpose(make_not_null(z_deriv_component.data()),
                      u_component.data(), n_r, n_phi * n_z);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), z_deriv_component.data(), diff_matrix_phi,
            num_grid_points);
        raw_transpose(make_not_null(phi_deriv_component.data()), buffer->data(),
                      n_phi * n_z, n_r);

        // See comments in disk_apply() in the .tpp for considerations about
        // the following code. For this function, we don't have enough buffer
        // space to cleanly use the approach in the .tpp, so we use the
        // masking approach instead. Note that timing shows this to only be a
        // few percent slower

        // reuse transposed data to go to angular spectral space
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), z_deriv_component.data(),
            nodal_to_modal_phi, num_grid_points);
        // r_deriv_component holds transposed angular modal rep

        raw_transpose(make_not_null(buffer->data()), r_deriv_component.data(),
                      n_phi * n_z, n_r);
        std::copy(buffer->data(), buffer->data() + num_grid_points,
                  z_deriv_component.data());
        // buffer and z_deriv_component hold angular modal rep

        // buffer stores even components, z_deriv_component stores odd
        for (size_t k = 0; k < n_z; ++k) {
          size_t offset = k * n_phi * n_r;
          for (size_t i = 0; i < n_phi; ++i) {
            if (i == 0) {
              // This is an even column -> zero odd vals
              std::fill(z_deriv_component.data() + offset,
                        z_deriv_component.data() + offset + n_r, 0.0);
            } else {
              if ((i - 1) / 2 % 2 == 1) {
                // This is an even column with even next to it -> zero odd vals
                std::fill(z_deriv_component.data() + offset,
                          z_deriv_component.data() + offset + 2 * n_r, 0.0);
              } else {
                // This is an even column with odd next to it -> zero even vals
                std::fill(buffer->data() + offset,
                          buffer->data() + offset + 2 * n_r, 0.0);
              }
              ++i;
              offset += n_r;
            }
            offset += n_r;
          }
        }
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), buffer->data(), diff_matrix_r_even,
            num_grid_points, false);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), z_deriv_component.data(),
            diff_matrix_r_odd, num_grid_points, true);
        // r_deriv_component holds the radial derivative in angular modal rep

        raw_transpose(make_not_null(buffer->data()), r_deriv_component.data(),
                      n_r, n_phi * n_z);
        // buffer holds transposed radial derivative in angular modal rep

        partial_derivatives_detail::apply_matrix_in_first_dim(
            z_deriv_component.data(), buffer->data(), modal_to_nodal_phi,
            num_grid_points);
        // z_deriv_component holds transposed radial derivative in angular nodal
        // rep

        raw_transpose(make_not_null(r_deriv_component.data()),
                      z_deriv_component.data(), n_phi * n_z, n_r);

        // z derivative
        raw_transpose(make_not_null(z_deriv_component.data()),
                      u_component.data(), n_r * n_phi, n_z);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), z_deriv_component.data(), diff_matrix_z,
            num_grid_points);
        raw_transpose(make_not_null(z_deriv_component.data()), buffer->data(),
                      n_z, n_r * n_phi);
      }
    } else {
      ERROR(
          "Support for complex numbers with cylinder is not yet implemented "
          "for logical_partial_derivative.");
    }
  } else {
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
      partial_derivatives_detail::apply_matrix_in_first_dim(
          // NOLINTNEXTLINE(readability-redundant-smartptr-get)
          logical_derivative_of_u->get(xi_deriv_tensor_index).data(),
          u[storage_index].data(), diff_matrices[0].get(), num_grid_points);
      for (size_t i = 1; i < Dim; ++i) {
        const auto deriv_tensor_index = prepend(u_tensor_index, i);
        DataType& deriv_component =
            logical_derivative_of_u->get(deriv_tensor_index);
        size_t chunk_size =
            diff_matrices[0].get().rows() *
            (i == 1 ? 1 : gsl::at(diff_matrices, 1).get().rows());
        raw_transpose(make_not_null(deriv_component.data()),
                      u[storage_index].data(), chunk_size,
                      num_grid_points / chunk_size);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), deriv_component.data(),
            gsl::at(diff_matrices, i).get(), num_grid_points);
        chunk_size =
            i == 1 ? (Dim == 2 ? gsl::at(diff_matrices, 1).get().rows()
                               : gsl::at(diff_matrices, 1).get().rows() *
                                     gsl::at(diff_matrices, 2).get().rows())
                   : gsl::at(diff_matrices, 2).get().rows();
        raw_transpose(make_not_null(deriv_component.data()), buffer->data(),
                      chunk_size, num_grid_points / chunk_size);
      }
    }
  }
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  using ValueType = typename DataType::value_type;  // double or complex<double>
  std::vector<ValueType> buffer(mesh.number_of_grid_points());
  gsl::span<ValueType> buffer_view{buffer.data(), buffer.size()};
  logical_partial_derivative(logical_derivative_of_u,
                             make_not_null(&buffer_view), u, mesh);
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
auto logical_partial_derivative(const Tensor<DataType, SymmList, IndexList>& u,
                                const Mesh<Dim>& mesh)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      result{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&result), u, mesh);
  return result;
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>*>
        du,
    const TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>& logical_partial_derivative_of_u,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  for (size_t storage_index = 0;
       storage_index < Tensor<DataType, SymmList, IndexList>::size();
       ++storage_index) {
    const auto u_multi_index =
        Tensor<DataType, SymmList,
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

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>*>
        du,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      logical_partial_derivative_of_u{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&logical_partial_derivative_of_u), u,
                             mesh);
  partial_derivative<DataType, SymmList, IndexList>(
      du, logical_partial_derivative_of_u, inverse_jacobian);
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
auto partial_derivative(
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>
      result{mesh.number_of_grid_points()};
  partial_derivative(make_not_null(&result), u, mesh, inverse_jacobian);
  return result;
}

#define GET_DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define GET_TENSOR(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
                        GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),     \
                        GET_FRAME(data)>,                                      \
                    GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >         \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<typename GET_DTYPE(data)::value_type>*> buffer,  \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
                        GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),     \
                        GET_FRAME(data)>,                                      \
                    GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >         \
          logical_derivative_of_u,                                             \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>,     \
      GET_DIM(data), UpLo::Lo,                                                 \
      Frame::ElementLogical >                                                  \
          logical_partial_derivative(const GET_TENSOR(data) < GET_DTYPE(data), \
                                     GET_DIM(data), GET_FRAME(data) > &u,      \
                                     const Mesh<GET_DIM(data)>& mesh);         \
  template void                                                                \
      partial_derivative<GET_DTYPE(data), GET_TENSOR(data) < GET_DTYPE(data),  \
                         GET_DIM(data), GET_FRAME(data)>::symmetry,            \
      GET_TENSOR(                                                              \
          data)<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>::index_list > \
          (const gsl::not_null<TensorMetafunctions::prepend_spatial_index<     \
                                   GET_TENSOR(data) < GET_DTYPE(data),         \
                                   GET_DIM(data), GET_FRAME(data)>,            \
                               GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* >    \
               du,                                                             \
           const TensorMetafunctions::prepend_spatial_index<                   \
               GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),              \
               GET_FRAME(data)>,                                               \
           GET_DIM(data), UpLo::Lo,                                            \
           Frame::ElementLogical > &logical_partial_derivative_of_u,           \
           const InverseJacobian<DataVector, GET_DIM(data),                    \
                                 Frame::ElementLogical, GET_FRAME(data)>       \
               & inverse_jacobian);                                            \
  template void partial_derivative(                                            \
      const gsl::not_null<TensorMetafunctions::prepend_spatial_index<          \
                              GET_TENSOR(data) < GET_DTYPE(data),              \
                              GET_DIM(data), GET_FRAME(data)>,                 \
                          GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* > du,     \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,                   \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>,     \
      GET_DIM(data), UpLo::Lo,                                                 \
      GET_FRAME(data) >                                                        \
          partial_derivative(                                                  \
              const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),         \
              GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,           \
              const InverseJacobian<DataVector, GET_DIM(data),                 \
                                    Frame::ElementLogical, GET_FRAME(data)>&   \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (tnsr::a, tnsr::A, tnsr::i, tnsr::I, tnsr::ab, tnsr::Ab,
                         tnsr::aB, tnsr::AB, tnsr::ij, tnsr::iJ, tnsr::Ij,
                         tnsr::IJ, tnsr::iA, tnsr::ia, tnsr::aa, tnsr::AA,
                         tnsr::ii, tnsr::II, tnsr::ijj, tnsr::Ijj, tnsr::iaa))

#undef INSTANTIATION

// Some additional mixed-dimension instantiations
template TensorMetafunctions::prepend_spatial_index<
    tnsr::aa<ComplexDataVector, 3, Frame::Inertial>, 2, UpLo::Lo,
    Frame::Inertial>
partial_derivative(const tnsr::aa<ComplexDataVector, 3, Frame::Inertial>& u,
                   const Mesh<2>& mesh,
                   const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                                         Frame::Inertial>& inverse_jacobian);

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
          Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo,                    \
          Frame::ElementLogical>*>                                             \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<typename GET_DTYPE(data)::value_type>*> buffer,  \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh);      \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
          Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo,                    \
          Frame::ElementLogical>*>                                             \
          logical_derivative_of_u,                                             \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh);      \
  template TensorMetafunctions::prepend_spatial_index<                         \
      Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo, Frame::ElementLogical> \
  logical_partial_derivative(const Scalar<GET_DTYPE(data)>& u,                 \
                             const Mesh<GET_DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATE_JACOBIANS(r, data)                                       \
  template TensorMetafunctions::prepend_spatial_index<                       \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                     \
      Frame::ElementLogical, GET_FRAME(data)>,                               \
      GET_DIM(data), UpLo::Lo,                                               \
      GET_FRAME(data) >                                                      \
          partial_derivative(                                                \
              const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),       \
              Frame::ElementLogical, GET_FRAME(data) > &u,                   \
              const Mesh<GET_DIM(data)>& mesh,                               \
              const InverseJacobian<DataVector, GET_DIM(data),               \
                                    Frame::ElementLogical, GET_FRAME(data)>& \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIANS, (DataVector), (1, 2, 3),
                        (Frame::Inertial), (InverseJacobian))

#undef INSTANTIATE_JACOBIANS

#define INSTANTIATE_SCALAR(r, data)                                            \
  template void partial_derivative<GET_DTYPE(data), Symmetry<>, index_list<>>( \
      gsl::not_null<tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>*> \
          du,                                                                  \
      const tnsr::i<GET_DTYPE(data), GET_DIM(data), Frame::ElementLogical>&    \
          logical_partial_derivative_of_u,                                     \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template void partial_derivative(                                            \
      gsl::not_null<tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>*> \
          du,                                                                  \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh,       \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>            \
  partial_derivative(                                                          \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh,       \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR, (DataVector, ComplexDataVector),
                        (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef INSTANTIATE_SCALAR
#undef GET_FRAME
#undef GET_DIM
#undef GET_TENSOR
