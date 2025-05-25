// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IrregularInterpolant.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/CardinalInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ContainerHelpers.hpp"

namespace {

// Just linear for now, can be extended to higher order...
// Future optimization: it might be more efficient to use Blaze's sparse
// matrices since the interpolation matrix is mostly zeros
std::vector<double> fd_stencil(const DataVector& xi_source,
                               const double xi_target) {
  ASSERT(std::is_sorted(std::begin(xi_source), std::end(xi_source)),
         "xi_source = " << xi_source);
  auto xi_u =
      std::upper_bound(std::begin(xi_source), std::end(xi_source), xi_target);
  if (std::end(xi_source) == xi_u) {
    std::advance(xi_u, -1);
  }
  if (std::begin(xi_source) == xi_u) {
    std::advance(xi_u, 1);
  }
  const auto xi_l = std::prev(xi_u);
  auto index = std::distance(std::begin(xi_source), xi_l);
  std::vector<double> result(xi_source.size(), 0.0);
  const auto result_l = std::next(std::begin(result), index);
  const auto result_u = std::next(result_l, 1);
  *result_l = (*xi_u - xi_target) / (*xi_u - *xi_l);
  *result_u = (xi_target - *xi_l) / (*xi_u - *xi_l);
  return result;
}

template <size_t Dim, typename DataType>
Matrix interpolation_matrix(
    const Mesh<Dim>& mesh,
    const tnsr::I<DataType, Dim, Frame::ElementLogical>& points);

template <typename DataType>
Matrix interpolation_matrix(
    const Mesh<1>& mesh,
    const tnsr::I<DataType, 1, Frame::ElementLogical>& points) {
  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    auto source_xi = logical_coordinates(mesh);
    const auto number_of_source_points = mesh.number_of_grid_points();
    const DataVector xi_source(get<0>(source_xi).data(),
                               number_of_source_points);
    const size_t number_of_target_points = get_size(get<0>(points));
    Matrix result(number_of_target_points, number_of_source_points);
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get_element(get<0>(points), p);
      const auto stencil = fd_stencil(xi_source, xi_target);
      for (size_t i = 0; i < number_of_source_points; ++i) {
        result(p, i) = stencil[i];
      }
    }
    return result;
  }

  // Not FD, so use spectral interpolation
  return Spectral::interpolation_matrix(mesh, get<0>(points));
}

template <typename DataType>
Matrix interpolation_matrix(
    const Mesh<2>& mesh,
    const tnsr::I<DataType, 2, Frame::ElementLogical>& points) {
  const auto number_of_target_points = get_size(get<0>(points));
  Matrix result(number_of_target_points, mesh.number_of_grid_points());

  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::FiniteDifference,
           "Mixed FD and DG bases are not supported. Mesh = " << mesh);
    auto source_xi = logical_coordinates(mesh);
    DataVector xi_source{get<0>(source_xi).data(), mesh.extents(0)};
    DataVector eta_source;
    if (mesh.extents(1) == mesh.extents(0)) {
      eta_source.set_data_ref(&xi_source);
    } else {
      eta_source.destructive_resize(mesh.extents(1));
      for (size_t j = 0; j < mesh.extents(1); ++j) {
        eta_source[j] = get<1>(source_xi)[j * mesh.extents(0)];
      }
    }
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get_element(get<0>(points), p);
      const double eta_target = get_element(get<1>(points), p);
      const auto xi_stencil = fd_stencil(xi_source, xi_target);
      const auto eta_stencil = fd_stencil(eta_source, eta_target);
      for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
        for (size_t i = 0; i < mesh.extents(0); ++i) {
          result(p, s) = xi_stencil[i] * eta_stencil[j];
          ++s;
        }
      }
    }
    return result;
  } else if (mesh.basis()[0] == Spectral::Basis::SphericalHarmonic) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::SphericalHarmonic,
           "Expected both dimensions to have spherical harmonic basis. Mesh = "
               << mesh);
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto [n_th, n_ph] = mesh.extents().indices();
    const auto& [m_th, m_ph] = cardinal_interpolator.interpolation_matrices();
    for (size_t i_ph = 0, s = 0; i_ph < n_ph; ++i_ph) {
      for (size_t i_th = 0; i_th < n_th; ++i_th) {
        for (size_t k = 0; k < number_of_target_points; ++k) {
          result(k, s) = m_th(k, i_th) * m_ph(2 * k, i_ph) +
                         m_th(k, i_th + n_th) * m_ph(2 * k + 1, i_ph);
        }
        ++s;
      }
    }
    return result;
  }

  // Not FD or special basis, so use 1D spectral interpolation matrices
  const std::array<Matrix, 2> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points))}};

  // First dimension of DataVector varies fastest.
  for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
    for (size_t i = 0; i < mesh.extents(0); ++i) {
      for (size_t p = 0; p < number_of_target_points; ++p) {
        result(p, s) = matrices[0](p, i) * matrices[1](p, j);
      }
      ++s;
    }
  }
  return result;
}

template <typename DataType>
Matrix interpolation_matrix(
    const Mesh<3>& mesh,
    const tnsr::I<DataType, 3, Frame::ElementLogical>& points) {
  const auto number_of_target_points = get_size(get<0>(points));
  Matrix result(number_of_target_points, mesh.number_of_grid_points());

  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::FiniteDifference and
               mesh.basis()[2] == Spectral::Basis::FiniteDifference,
           "Mixed FD and DG bases are not supported. Mesh = " << mesh);
    auto source_xi = logical_coordinates(mesh);
    DataVector xi_source{get<0>(source_xi).data(), mesh.extents(0)};
    DataVector eta_source;
    if (mesh.extents(1) == mesh.extents(0)) {
      eta_source.set_data_ref(&xi_source);
    } else {
      eta_source.destructive_resize(mesh.extents(1));
      for (size_t j = 0; j < mesh.extents(1); ++j) {
        eta_source[j] = get<1>(source_xi)[j * mesh.extents(0)];
      }
    }
    DataVector zeta_source;
    if (mesh.extents(2) == mesh.extents(0)) {
      zeta_source.set_data_ref(&xi_source);
    } else if (mesh.extents(2) == mesh.extents(1)) {
      zeta_source.set_data_ref(&eta_source);
    } else {
      zeta_source.destructive_resize(mesh.extents(2));
      for (size_t k = 0; k < mesh.extents(2); ++k) {
        zeta_source[k] =
            get<2>(source_xi)[k * mesh.extents(0) * mesh.extents(1)];
      }
    }
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get_element(get<0>(points), p);
      const double eta_target = get_element(get<1>(points), p);
      const double zeta_target = get_element(get<2>(points), p);
      const auto xi_stencil = fd_stencil(xi_source, xi_target);
      const auto eta_stencil = fd_stencil(eta_source, eta_target);
      const auto zeta_stencil = fd_stencil(zeta_source, zeta_target);
      for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
        for (size_t j = 0; j < mesh.extents(1); ++j) {
          for (size_t i = 0; i < mesh.extents(0); ++i) {
            result(p, s) = xi_stencil[i] * eta_stencil[j] * zeta_stencil[k];
            ++s;
          }
        }
      }
    }
    return result;
  } else if (mesh.basis()[1] == Spectral::Basis::SphericalHarmonic) {
    ASSERT(mesh.basis()[2] == Spectral::Basis::SphericalHarmonic,
           "Expected last two dimensions to each have spherical harmonic "
           "basis. Mesh = "
               << mesh);
    const intrp::Cardinal cardinal_interpolator(mesh, points);
    const auto [n_r, n_th, n_ph] = mesh.extents().indices();
    const auto& [m_r, m_th, m_ph] =
        cardinal_interpolator.interpolation_matrices();
    for (size_t i_ph = 0, s = 0; i_ph < n_ph; ++i_ph) {
      for (size_t i_th = 0; i_th < n_th; ++i_th) {
        for (size_t i_r = 0; i_r < n_r; ++i_r) {
          for (size_t k = 0; k < number_of_target_points; ++k) {
            result(k, s) =
                m_r(k, i_r) * (m_th(k, i_th) * m_ph(2 * k, i_ph) +
                               m_th(k, i_th + n_th) * m_ph(2 * k + 1, i_ph));
          }
          ++s;
        }
      }
    }
    return result;
  }

  // Not FD or special basis, so use 1D spectral interpolation matrices
  const std::array<Matrix, 3> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(2), get<2>(points))}};

  // First dimension of DataVector varies fastest.
  for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
    for (size_t j = 0; j < mesh.extents(1); ++j) {
      for (size_t i = 0; i < mesh.extents(0); ++i) {
        for (size_t p = 0; p < number_of_target_points; ++p) {
          result(p, s) =
              matrices[0](p, i) * matrices[1](p, j) * matrices[2](p, k);
        }
        ++s;
      }
    }
  }
  return result;
}
}  // namespace

namespace intrp {

template <size_t Dim>
Irregular<Dim>::Irregular() = default;

template <size_t Dim>
Irregular<Dim>::Irregular(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points)
    : interpolation_matrix_(interpolation_matrix(source_mesh, target_points)) {}

template <size_t Dim>
Irregular<Dim>::Irregular(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<double, Dim, Frame::ElementLogical>& target_point)
    : interpolation_matrix_(interpolation_matrix(source_mesh, target_point)) {}

template <size_t Dim>
void Irregular<Dim>::pup(PUP::er& p) {
  p | interpolation_matrix_;
}

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<DataVector*> result,
                                 const DataVector& input) const {
  const size_t m = interpolation_matrix_.rows();
  const size_t k = interpolation_matrix_.columns();
  ASSERT(k == input.size(),
         "Number of points in 'input', "
             << input.size()
             << ",\n disagrees with the size of the source_mesh, " << k
             << ", that was passed into the constructor");
  if (result->size() != m) {
    result->destructive_resize(m);
  }
  dgemv_('n', m, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(), input.data(), 1, 0.0, result->data(),
         1);
}

template <size_t Dim>
DataVector Irregular<Dim>::interpolate(const DataVector& input) const {
  DataVector result{input.size()};
  interpolate(make_not_null(&result), input);
  return result;
}

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<ComplexDataVector*> result,
                                 const ComplexDataVector& input) const {
  const size_t m = interpolation_matrix_.rows();
  const size_t k = interpolation_matrix_.columns();
  ASSERT(k == input.size(),
         "Number of points in 'input', "
             << input.size()
             << ",\n disagrees with the size of the source_mesh, " << k
             << ", that was passed into the constructor");
  if (result->size() != m) {
    result->destructive_resize(m);
  }
  // Possible performance optimization: can possibly be written as a single
  // `dgemm` call, or might be faster using `zgemv`.
  // NOLINTBEGIN
  dgemv_('n', m, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(),
         reinterpret_cast<const double*>(input.data()), 2, 0.0,
         reinterpret_cast<double*>(result->data()), 2);
  dgemv_('n', m, k, 1.0, interpolation_matrix_.data(),
         interpolation_matrix_.spacing(),
         reinterpret_cast<const double*>(input.data()) + 1, 2, 0.0,
         reinterpret_cast<double*>(result->data()) + 1, 2);
  // NOLINTEND
}

template <size_t Dim>
ComplexDataVector Irregular<Dim>::interpolate(
    const ComplexDataVector& input) const {
  ComplexDataVector result{input.size()};
  interpolate(make_not_null(&result), input);
  return result;
}

namespace {

template <typename ValueType>
void span_interpolate_impl(const gsl::not_null<gsl::span<ValueType>*> result,
                           const gsl::span<const ValueType>& input,
                           const Matrix& interpolation_matrix) {
  const size_t m = interpolation_matrix.rows();
  const size_t k = interpolation_matrix.columns();
  ASSERT(input.size() % k == 0,
         "Number of points in 'input', "
             << input.size()
             << ",\n must be a multiple of the source grid points, " << k
             << ", that was passed into the constructor");
  const size_t number_of_components = input.size() / k;
  ASSERT(result->size() == number_of_components * m,
         "The result must be of size " << number_of_components * m
                                       << " but got " << result->size());
  if constexpr (std::is_same_v<ValueType, double>) {
    dgemm_<true>('N', 'N', m, number_of_components, k, 1.0,
                 interpolation_matrix.data(), interpolation_matrix.spacing(),
                 input.data(), k, 0.0, result->data(), m);
  } else if constexpr (std::is_same_v<ValueType, float>) {
    // No BLAS function exists for mixed precision so we do the matrix multiply
    // manually. A possible performance optimization would be to compute the
    // interpolation matrix in single precision and then use sgemm.
    for (size_t j = 0; j < number_of_components; ++j) {
      for (size_t i = 0; i < m; ++i) {
        float sum = interpolation_matrix.data()[i] * input[j * k];
        for (size_t l = 1; l < k; ++l) {
          sum += interpolation_matrix.data()[i + l * m] * input[l + j * k];
        }
        (*result)[i + j * m] = sum;
      }
    }
  } else if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
    // BLAS zgemm operates on complex matrices, so we need to copy the real
    // matrix to a complex matrix with zero imaginary part before calling zgemm.
    // Note by Nils Vu (Aug 2024): Profiling of partial derivatives showed that
    // this zgemm approach with a complex matrix is still faster than
    // transposing the complex input data and applying the real matrix with
    // dgemm (though maybe the transpose can be avoided with smarter dgemm
    // strides). Possible performance optimization: avoid the copy here by
    // caching the complex matrix.
    const blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>
        matrix_complex{interpolation_matrix};
    zgemm_<true>('N', 'N', m, number_of_components, k,
                 std::complex<double>{1.0, 0.0}, matrix_complex.data(),
                 matrix_complex.spacing(), input.data(), k,
                 std::complex<double>{0.0, 0.0}, result->data(), m);
  }
}
}  // namespace

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<gsl::span<double>*> result,
                                 const gsl::span<const double>& input) const {
  span_interpolate_impl(result, input, interpolation_matrix_);
}

template <size_t Dim>
void Irregular<Dim>::interpolate(
    const gsl::not_null<gsl::span<std::complex<double>>*> result,
    const gsl::span<const std::complex<double>>& input) const {
  span_interpolate_impl(result, input, interpolation_matrix_);
}

template <size_t Dim>
void Irregular<Dim>::interpolate(const gsl::not_null<gsl::span<float>*> result,
                                 const gsl::span<const float>& input) const {
  span_interpolate_impl(result, input, interpolation_matrix_);
}

template <size_t Dim>
bool operator!=(const Irregular<Dim>& lhs, const Irregular<Dim>& rhs) {
  return not(lhs == rhs);
}

template class Irregular<1>;
template class Irregular<2>;
template class Irregular<3>;
template bool operator!=(const Irregular<1>& lhs, const Irregular<1>& rhs);
template bool operator!=(const Irregular<2>& lhs, const Irregular<2>& rhs);
template bool operator!=(const Irregular<3>& lhs, const Irregular<3>& rhs);

}  // namespace intrp
