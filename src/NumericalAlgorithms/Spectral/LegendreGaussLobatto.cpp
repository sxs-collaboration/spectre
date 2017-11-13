// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"

namespace Basis {
namespace lgl {
namespace detail {
class LglQp {
 public:
  /// See Algorithm 24 from Kopriva's book, p. 65 and the surrounding discussion
  /// for details
  LglQp(size_t poly_degree, double x);

  double q() const noexcept { return q_; }
  double q_prime() const noexcept { return q_prime_; }
  double p() const noexcept { return p_; }

 private:
  double q_;
  double q_prime_;
  double p_;
};

LglQp::LglQp(const size_t poly_degree, const double x) {
  // Algorithm 24 from Kopriva's book, p. 65
  // Note: Book has errors in last 4 lines. Corrected in errata on website
  ASSERT(poly_degree > 1, "polynomial degree must be greater than one");

  // Evaluate P_n(x), q(x) = P_(n+1) - P_(n-1), and q'(x) for n >= 2
  double p_n_minus_2 = 1.0;
  double p_n_minus_1 = x;
  double p_prime_n_minus_2 = 0.0;
  double p_prime_n_minus_1 = 1.0;
  double p_n = std::numeric_limits<double>::signaling_NaN();
  for (size_t k = 2; k <= poly_degree; k++) {
    // recurrence relation
    p_n = ((2 * k - 1) * x * p_n_minus_1 - (k - 1) * p_n_minus_2) / k;
    const double p_prime_n = p_prime_n_minus_2 + (2 * k - 1) * p_n_minus_1;
    p_n_minus_2 = p_n_minus_1;
    p_n_minus_1 = p_n;
    p_prime_n_minus_2 = p_prime_n_minus_1;
    p_prime_n_minus_1 = p_prime_n;
  }
  const size_t k = poly_degree + 1;
  const double p_n_plus_1 = ((2 * k - 1) * x * p_n - (k - 1) * p_n_minus_2) / k;
  const double p_prime_n_plus_1 = p_prime_n_minus_2 + (2 * k - 1) * p_n_minus_1;

  q_ = p_n_plus_1 - p_n_minus_2;
  q_prime_ = p_prime_n_plus_1 - p_prime_n_minus_2;
  p_ = p_n;
}

class CollocationPointsAndWeights {
 public:
  explicit CollocationPointsAndWeights(size_t number_of_pts);

  const DataVector& collocation_pts() const noexcept {
    return collocation_pts_;
  }
  const DataVector& quadrature_weights() const noexcept {
    return quadrature_weights_;
  }
  const DataVector& barycentric_weights() const noexcept {
    return barycentric_weights_;
  }

 private:
  DataVector collocation_pts_;
  DataVector quadrature_weights_;
  DataVector barycentric_weights_;
};

CollocationPointsAndWeights::CollocationPointsAndWeights(
    const size_t number_of_pts)
    : collocation_pts_(number_of_pts),
      quadrature_weights_(number_of_pts),
      barycentric_weights_(number_of_pts) {
  // Algorithm 25 from Kopriva's book, p.66
  ASSERT(number_of_pts > 1, "Must have more than one collocation point");
  size_t poly_degree = number_of_pts - 1;

  collocation_pts_[0] = -1.0;
  collocation_pts_[poly_degree] = 1.0;
  quadrature_weights_[0] = quadrature_weights_[poly_degree] =
      2.0 / (poly_degree * (poly_degree + 1));
  const size_t maxit = 50;
  constexpr double tolerance = 4.0 * std::numeric_limits<double>::epsilon();
  for (size_t j = 1; j < (poly_degree + 1) / 2; j++) {
    // initial guess for Newton-Raphson iteration:
    double logical_coord = -cos((j + 0.25) * M_PI / poly_degree -
                                0.375 / (poly_degree * M_PI * (j + 0.25)));
    size_t iteration = 0;
    double delta;
    do {
      const LglQp q_and_p(poly_degree, logical_coord);
      delta = -q_and_p.q() / q_and_p.q_prime();
      logical_coord += delta;
      iteration++;
      if (iteration > maxit) {
        // LCOV_EXCL_START
        ERROR(
            "Legendre-Gauss-Lobatto computing collocation points exceeded "
            "maximum number of iterations ("
            << maxit << ") \n");
        // LCOV_EXCL_STOP
      }
    } while (std::abs(delta) > tolerance * std::abs(logical_coord));

    const LglQp q_and_p(poly_degree, logical_coord);
    collocation_pts_[j] = logical_coord;
    collocation_pts_[poly_degree - j] = -logical_coord;
    quadrature_weights_[j] = quadrature_weights_[poly_degree - j] =
        2.0 / (poly_degree * (poly_degree + 1) * q_and_p.p() * q_and_p.p());
  }

  if (poly_degree % 2 == 0) {
    // The origin (0.0) is a collocation point if poly_degree (N) is even
    const LglQp q_and_p(poly_degree, 0.0);
    collocation_pts_[poly_degree / 2] = 0.0;
    quadrature_weights_[poly_degree / 2] =
        2.0 / (poly_degree * (poly_degree + 1) * q_and_p.p() * q_and_p.p());
  }

  // use closed form expression for Legendre-Gauss-Lobatto barycentric weights
  for (size_t i = 0; i < number_of_pts; i++) {
    barycentric_weights_[i] = i % 2 == 0 ? sqrt(quadrature_weights_[i])
                                         : -sqrt(quadrature_weights_[i]);
  }
}

const CollocationPointsAndWeights& collocation_points_and_weights(
    const size_t number_of_pts) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= ::Basis::lgl::maximum_number_of_pts,
         "Exceeded maximum number of collocation points.");
  static const auto collo_pts_and_weights = []() {
    std::vector<CollocationPointsAndWeights> local_collo_pts;
    local_collo_pts.reserve(::Basis::lgl::maximum_number_of_pts - 1);
    for (size_t n = 2; n <= Basis::lgl::maximum_number_of_pts; ++n) {
      local_collo_pts.emplace_back(n);
    }
    return local_collo_pts;
  }();
  return collo_pts_and_weights[number_of_pts - 2];
}

Matrix differentiation_matrix(const size_t number_of_pts) {
  // This implements algorithm 37 on p.82 of Kopriva's book. It is valid for any
  // collocation points and barycentricx weights
  const DataVector& collocation_pts =
      collocation_points_and_weights(number_of_pts).collocation_pts();
  const DataVector& bary_weights =
      collocation_points_and_weights(number_of_pts).barycentric_weights();

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

Matrix spectral_to_grid_points_matrix(const size_t number_of_pts) {
  // Since u(x) = sum a_k P_k(x), matrix for u(x_j) is just P_k(x_j) transposed
  const DataVector& collocation_pts =
      collocation_points_and_weights(number_of_pts).collocation_pts();
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
Matrix grid_points_to_spectral_matrix(const size_t number_of_pts) {
  const DataVector& x =
      collocation_points_and_weights(number_of_pts).collocation_pts();
  const DataVector& w =
      collocation_points_and_weights(number_of_pts).quadrature_weights();
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
}  // namespace detail

const DataVector& collocation_points(const size_t number_of_pts) {
  return detail::collocation_points_and_weights(number_of_pts)
      .collocation_pts();
}

const DataVector& quadrature_weights(const size_t number_of_pts) {
  return detail::collocation_points_and_weights(number_of_pts)
      .quadrature_weights();
}

const Matrix& differentiation_matrix(const size_t number_of_pts) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= ::Basis::lgl::maximum_number_of_pts,
         "Exceeded maximum number of collocation points.");
  static const auto differentiation_matrices = []() {
    std::vector<Matrix> local_diff_matrices;
    local_diff_matrices.reserve(::Basis::lgl::maximum_number_of_pts - 1);
    for (size_t n = 2; n <= ::Basis::lgl::maximum_number_of_pts; ++n) {
      local_diff_matrices.emplace_back(detail::differentiation_matrix(n));
    }
    return local_diff_matrices;
  }();
  return differentiation_matrices[number_of_pts - 2];
}

const Matrix& grid_points_to_spectral_matrix(const size_t number_of_pts) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= ::Basis::lgl::maximum_number_of_pts,
         "Exceeded maximum number of collocation points.");
  static const auto grid_points_to_spectral_matrices = []() {
    std::vector<Matrix> local_grid_to_spec_matrices;
    local_grid_to_spec_matrices.reserve(::Basis::lgl::maximum_number_of_pts -
                                        1);
    for (size_t n = 2; n <= ::Basis::lgl::maximum_number_of_pts; ++n) {
      local_grid_to_spec_matrices.emplace_back(
          detail::grid_points_to_spectral_matrix(n));
    }
    return local_grid_to_spec_matrices;
  }();
  return grid_points_to_spectral_matrices[number_of_pts - 2];
}

const Matrix& spectral_to_grid_points_matrix(const size_t number_of_pts) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= ::Basis::lgl::maximum_number_of_pts,
         "Exceeded maximum number of collocation points.");
  static const auto spectral_to_grid_matrices = []() {
    std::vector<Matrix> local_spec_to_grid_matrices;
    local_spec_to_grid_matrices.reserve(::Basis::lgl::maximum_number_of_pts -
                                        1);
    for (size_t n = 2; n <= ::Basis::lgl::maximum_number_of_pts; ++n) {
      local_spec_to_grid_matrices.emplace_back(
          detail::spectral_to_grid_points_matrix(n));
    }
    return local_spec_to_grid_matrices;
  }();
  return spectral_to_grid_matrices[number_of_pts - 2];
}

const Matrix& linear_filter_matrix(const size_t number_of_pts) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= ::Basis::lgl::maximum_number_of_pts,
         "Exceeded maximum number of collocation points.");
  static const auto linear_filter_matrices = []() {
    std::vector<Matrix> local_linear_filter_matrices;
    local_linear_filter_matrices.reserve(::Basis::lgl::maximum_number_of_pts -
                                         1);
    for (size_t n = 2; n <= ::Basis::lgl::maximum_number_of_pts; ++n) {
      Matrix filter_matrix(n, n);
      // Linear filter is
      // grid_to_spectral_matrix * diag(1,1,0,0,....) * spectral_to_grid_matrix,
      // which multiplies first two columns of g2s with the first two rows of
      // spectral_to_grid_matrix
      dgemm_('N', 'N', n, n, 2, 1.0, spectral_to_grid_points_matrix(n).data(),
             n, grid_points_to_spectral_matrix(n).data(), n, 0.0,
             filter_matrix.data(), n);
      local_linear_filter_matrices.emplace_back(std::move(filter_matrix));
    }
    return local_linear_filter_matrices;
  }();
  return linear_filter_matrices[number_of_pts - 2];
}

template <typename T>
Matrix interpolation_matrix(const size_t number_of_pts,
                            const T& target_points) {
  ASSERT(number_of_pts > 1, "Must have at least two collocation points");
  ASSERT(number_of_pts <= ::Basis::lgl::maximum_number_of_pts,
         "Exceeded maximum number of collocation points.");
  const DataVector& collocation_pts =
      detail::collocation_points_and_weights(number_of_pts).collocation_pts();
  const DataVector& barycentric_weights =
      detail::collocation_points_and_weights(number_of_pts)
          .barycentric_weights();

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

}  // namespace lgl
}  // namespace Basis

/// \cond
template Matrix Basis::lgl::interpolation_matrix(
    const size_t num_collocation_points, const DataVector& target_points);
template Matrix Basis::lgl::interpolation_matrix(
    const size_t num_collocation_points,
    const std::vector<double>& target_points);
/// \endcond
