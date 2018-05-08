// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {

// Forward declarations with basis-specific implementations

/// \cond
/*!
 * \brief Computes the function value of the basis function \f$\Phi_k(x)\f$
 * (zero-indexed).
 */
template <Basis BasisType>
double compute_basis_function_value(size_t k, double x) noexcept;

/*!
 * \brief Computes the normalization square of the basis function \f$\Phi_k\f$
 * (zero-indexed), i.e. the definite integral over its square.
 */
template <Basis BasisType>
double compute_basis_function_normalization_square(size_t k) noexcept;

/*!
 * \brief Computes the collocation points and integral weights associated to the
 * basis and quadrature.
 */
template <Basis BasisType, Quadrature QuadratureType>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights(
    size_t num_points) noexcept;
/// \endcond

namespace {

const std::pair<DataVector, DataVector>& collocation_points_and_weights(
    const size_t number_of_pts) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= Spectral::maximum_number_of_points<Basis::Legendre>,
         "Exceeded maximum number of collocation points.");
  static const auto collo_pts_and_weights = []() {
    std::vector<std::pair<DataVector, DataVector>> local_collo_pts;
    local_collo_pts.reserve(
        Spectral::maximum_number_of_points<Basis::Legendre> - 1);
    for (size_t n = 2; n <= Spectral::maximum_number_of_points<Basis::Legendre>;
         ++n) {
      local_collo_pts.emplace_back(
          compute_collocation_points_and_weights<Basis::Legendre,
                                                 Quadrature::GaussLobatto>(n));
    }
    return local_collo_pts;
  }();
  return collo_pts_and_weights[number_of_pts - 2];
}

DataVector compute_barycentric_weights(const DataVector& x) noexcept {
  const size_t num_points = x.size();
  // This implements algorithm 30 on p. 75 of Kopriva's book.
  // It is valid for any collocation points.
  DataVector bary_weights(num_points, 1.);
  for (size_t j = 1; j < num_points; j++) {
    for (size_t k = 0; k < j; k++) {
      bary_weights[k] *= x[k] - x[j];
      bary_weights[j] *= x[j] - x[k];
    }
  }
  for (size_t j = 0; j < num_points; j++) {
    bary_weights[j] = 1. / bary_weights[j];
  }
  return bary_weights;
}

Matrix compute_differentiation_matrix(const size_t number_of_pts) {
  // This implements algorithm 37 on p.82 of Kopriva's book. It is valid for any
  // collocation points and barycentricx weights
  const DataVector& collocation_pts =
      collocation_points_and_weights(number_of_pts).first;
  DataVector bary_weights = compute_barycentric_weights(collocation_pts);

  Matrix diff_matrix(number_of_pts, number_of_pts);
  for (size_t i = 0; i < number_of_pts; ++i) {
    double diagonal = 0.0;
    for (size_t j = 0; j < number_of_pts; ++j) {
      if (LIKELY(i != j)) {
        diff_matrix(i, j) =
            bary_weights[j] /
            (bary_weights[i] * (collocation_pts[i] - collocation_pts[j]));
        diagonal -= diff_matrix(i, j);
      }
    }
    diff_matrix(i, i) = diagonal;
  }
  return diff_matrix;
}

Matrix compute_spectral_to_grid_points_matrix(const size_t number_of_pts) {
  // Since u(x) = sum a_k P_k(x), matrix for u(x_j) is just P_k(x_j) transposed
  const DataVector& collocation_pts =
      collocation_points_and_weights(number_of_pts).first;
  Matrix spec_to_grid(number_of_pts, number_of_pts);
  // number_of_pts >=2 ASSERT'd elsewhere
  for (size_t j = 0; j < number_of_pts; ++j) {
    spec_to_grid(j, 0) = 1.0;  // P_0 = 1
    double p_n_minus2 = 1.0;
    double p_n_minus1 = collocation_pts[j];  // P_1 = x
    spec_to_grid(j, 1) = p_n_minus1;
    for (size_t k = 2; k < number_of_pts; ++k) {
      const double p_n = ((2 * k - 1) * collocation_pts[j] * p_n_minus1 -
                          (k - 1) * p_n_minus2) /
                         k;
      spec_to_grid(j, k) = p_n;
      p_n_minus2 = p_n_minus1;
      p_n_minus1 = p_n;
    }
  }
  return spec_to_grid;
}

// GridPointsToSpectral is inverse matrix to above. Can compute numerically
// or use analytic expressions P_k(x_j) w_j/gamma_k where w_j=weight,
// gamma_k = normalization = 2/(2k+1), except = 2/N for k=N (for GLL)
Matrix compute_grid_points_to_spectral_matrix(const size_t number_of_pts) {
  const DataVector& x = collocation_points_and_weights(number_of_pts).first;
  const DataVector& w = collocation_points_and_weights(number_of_pts).second;
  Matrix g_to_s_matrix(number_of_pts, number_of_pts);
  for (size_t j = 0; j < number_of_pts; ++j) {
    g_to_s_matrix(0, j) = 0.5 * w[j];  // P_0 = 1
    // number_of_pts >=2 ASSERT'd elsewhere
    double p_n_minus2 = 1.0;
    double p_n_minus1 = x[j];  // P_1 = x
    g_to_s_matrix(1, j) =
        p_n_minus1 * w[j] * ((number_of_pts == 2) ? 0.5 : 1.5);
    for (size_t k = 2; k < number_of_pts; ++k) {
      const double p_n =
          ((2 * k - 1) * x[j] * p_n_minus1 - (k - 1) * p_n_minus2) / k;
      const double gamma = (k == number_of_pts - 1)
                               ? 2.0 / (number_of_pts - 1.0)
                               : 2.0 / (2 * k + 1);
      g_to_s_matrix(k, j) = p_n * w[j] / gamma;
      p_n_minus2 = p_n_minus1;
      p_n_minus1 = p_n;
    }
  }
  return g_to_s_matrix;
}
}  // namespace

template <>
const DataVector& collocation_points<Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t number_of_pts) noexcept {
  return collocation_points_and_weights(number_of_pts).first;
}

template <>
const DataVector& quadrature_weights<Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t number_of_pts) noexcept {
  return collocation_points_and_weights(number_of_pts).second;
}

template <>
const Matrix& differentiation_matrix<Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t number_of_pts) noexcept {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= Spectral::maximum_number_of_points<Basis::Legendre>,
         "Exceeded maximum number of collocation points.");
  static const auto differentiation_matrices = []() {
    std::vector<Matrix> local_diff_matrices;
    local_diff_matrices.reserve(
        Spectral::maximum_number_of_points<Basis::Legendre> - 1);
    for (size_t n = 2; n <= Spectral::maximum_number_of_points<Basis::Legendre>;
         ++n) {
      local_diff_matrices.emplace_back(compute_differentiation_matrix(n));
    }
    return local_diff_matrices;
  }();
  return differentiation_matrices[number_of_pts - 2];
}

template <>
const Matrix&
grid_points_to_spectral_matrix<Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t number_of_pts) noexcept {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= Spectral::maximum_number_of_points<Basis::Legendre>,
         "Exceeded maximum number of collocation points.");
  static const auto grid_points_to_spectral_matrices = []() {
    std::vector<Matrix> local_grid_to_spec_matrices;
    local_grid_to_spec_matrices.reserve(
        Spectral::maximum_number_of_points<Basis::Legendre> - 1);
    for (size_t n = 2; n <= Spectral::maximum_number_of_points<Basis::Legendre>;
         ++n) {
      local_grid_to_spec_matrices.emplace_back(
          compute_grid_points_to_spectral_matrix(n));
    }
    return local_grid_to_spec_matrices;
  }();
  return grid_points_to_spectral_matrices[number_of_pts - 2];
}

template <>
const Matrix&
spectral_to_grid_points_matrix<Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t number_of_pts) noexcept {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= Spectral::maximum_number_of_points<Basis::Legendre>,
         "Exceeded maximum number of collocation points.");
  static const auto spectral_to_grid_matrices = []() {
    std::vector<Matrix> local_spec_to_grid_matrices;
    local_spec_to_grid_matrices.reserve(
        Spectral::maximum_number_of_points<Basis::Legendre> - 1);
    for (size_t n = 2; n <= Spectral::maximum_number_of_points<Basis::Legendre>;
         ++n) {
      local_spec_to_grid_matrices.emplace_back(
          compute_spectral_to_grid_points_matrix(n));
    }
    return local_spec_to_grid_matrices;
  }();
  return spectral_to_grid_matrices[number_of_pts - 2];
}

template <>
const Matrix& linear_filter_matrix<Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t number_of_pts) noexcept {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= Spectral::maximum_number_of_points<Basis::Legendre>,
         "Exceeded maximum number of collocation points.");
  static const auto linear_filter_matrices = []() {
    std::vector<Matrix> local_linear_filter_matrices;
    local_linear_filter_matrices.reserve(
        Spectral::maximum_number_of_points<Basis::Legendre> - 1);
    for (size_t n = 2; n <= Spectral::maximum_number_of_points<Basis::Legendre>;
         ++n) {
      Matrix filter_matrix(n, n);
      // Linear filter is
      // grid_to_spectral_matrix * diag(1,1,0,0,....) * spectral_to_grid_matrix,
      // which multiplies first two columns of g2s with the first two rows of
      // spectral_to_grid_matrix
      dgemm_('N', 'N', n, n, 2, 1.0,
             spectral_to_grid_points_matrix<Basis::Legendre,
                                            Quadrature::GaussLobatto>(n)
                 .data(),
             n,
             grid_points_to_spectral_matrix<Basis::Legendre,
                                            Quadrature::GaussLobatto>(n)
                 .data(),
             n, 0.0, filter_matrix.data(), n);
      local_linear_filter_matrices.emplace_back(std::move(filter_matrix));
    }
    return local_linear_filter_matrices;
  }();
  return linear_filter_matrices[number_of_pts - 2];
}

template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(const size_t number_of_pts,
                            const T& target_points) noexcept {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= Spectral::maximum_number_of_points<Basis::Legendre>,
         "Exceeded maximum number of collocation points.");
  const DataVector& collocation_pts =
      collocation_points_and_weights(number_of_pts).first;
  DataVector barycentric_weights = compute_barycentric_weights(collocation_pts);

  const size_t num_target_points = target_points.size();
  Matrix interp_matrix(num_target_points, number_of_pts);
  const double eps = std::numeric_limits<double>::epsilon();

  for (size_t k = 0; k < num_target_points; ++k) {
    bool row_has_match = false;
    for (size_t j = 0; j < number_of_pts; ++j) {
      interp_matrix(k, j) = 0.0;
      if (((target_points[k] == 0.0 or collocation_pts[j] == 0.0) and
           std::abs(target_points[k] - collocation_pts[j]) <= 2.0 * eps) or
          (std::abs(target_points[k] - collocation_pts[j]) <=
               eps * std::abs(target_points[k]) and
           std::abs(target_points[k] - collocation_pts[j]) <=
               eps * std::abs(collocation_pts[j]))) {
        interp_matrix(k, j) = 1.0;
        row_has_match = true;
      }
    }
    if (not row_has_match) {
      double sum = 0.0;
      for (size_t j = 0; j < number_of_pts; ++j) {
        interp_matrix(k, j) =
            barycentric_weights[j] / (target_points[k] - collocation_pts[j]);
        sum += interp_matrix(k, j);
      }
      for (size_t j = 0; j < number_of_pts; ++j) {
        interp_matrix(k, j) /= sum;
      }
    }
  }
  return interp_matrix;
}

}  // namespace Spectral

/// \cond
template Matrix Spectral::interpolation_matrix<
    Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
    const size_t num_collocation_points,
    const DataVector& target_points) noexcept;
template Matrix Spectral::interpolation_matrix<
    Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
    const size_t num_collocation_points,
    const std::vector<double>& target_points) noexcept;
/// \endcond
