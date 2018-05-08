// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Spectral {

/// \cond
template <Basis, Quadrature>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights(
    size_t) noexcept;
/// \endcond

// Algorithms to compute Legendre-Gauss-Lobatto quadrature

namespace {

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

}  // namespace

/// \cond
template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t num_points) noexcept {
  DataVector collocation_pts(num_points);
  DataVector weights(num_points);

  // Algorithm 25 from Kopriva's book, p.66
  ASSERT(num_points > 1, "Must have more than one collocation point");
  size_t poly_degree = num_points - 1;

  collocation_pts[0] = -1.0;
  collocation_pts[poly_degree] = 1.0;
  weights[0] = weights[poly_degree] = 2.0 / (poly_degree * (poly_degree + 1));
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
    collocation_pts[j] = logical_coord;
    collocation_pts[poly_degree - j] = -logical_coord;
    weights[j] = weights[poly_degree - j] =
        2.0 / (poly_degree * (poly_degree + 1) * q_and_p.p() * q_and_p.p());
  }

  if (poly_degree % 2 == 0) {
    // The origin (0.0) is a collocation point if poly_degree (N) is even
    const LglQp q_and_p(poly_degree, 0.0);
    collocation_pts[poly_degree / 2] = 0.0;
    weights[poly_degree / 2] =
        2.0 / (poly_degree * (poly_degree + 1) * q_and_p.p() * q_and_p.p());
  }

  return std::make_pair(collocation_pts, weights);
}
/// \endcond

}  // namespace Spectral
