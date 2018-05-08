// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Spectral {

/// \cond
template <Basis, Quadrature>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights(
    size_t) noexcept;
/// \endcond

// Algorithms to compute Legendre-Gauss-Lobatto quadrature

namespace {

struct EvaluateQandL {
  EvaluateQandL(size_t poly_degree, double x) noexcept;
  double q;
  double q_prime;
  double L;
};

EvaluateQandL::EvaluateQandL(const size_t poly_degree,
                             const double x) noexcept {
  // Algorithm 24 from Kopriva's book, p. 65
  // Note: Book has errors in last 4 lines, corrected in errata on website
  // https://www.math.fsu.edu/~kopriva/publications/errata.pdf
  ASSERT(poly_degree >= 2, "Polynomial degree must be at least two.");
  double L_n_minus_2 = 1.;
  double L_n_minus_1 = x;
  double L_prime_n_minus_2 = 0.;
  double L_prime_n_minus_1 = 1.;
  double L_n = std::numeric_limits<double>::signaling_NaN();
  for (size_t k = 2; k <= poly_degree; k++) {
    L_n = ((2 * k - 1) * x * L_n_minus_1 - (k - 1) * L_n_minus_2) / k;
    const double L_prime_n = L_prime_n_minus_2 + (2 * k - 1) * L_n_minus_1;
    L_n_minus_2 = L_n_minus_1;
    L_n_minus_1 = L_n;
    L_prime_n_minus_2 = L_prime_n_minus_1;
    L_prime_n_minus_1 = L_prime_n;
  }
  const size_t k = poly_degree + 1;
  const double L_n_plus_1 = ((2 * k - 1) * x * L_n - (k - 1) * L_n_minus_2) / k;
  const double L_prime_n_plus_1 = L_prime_n_minus_2 + (2 * k - 1) * L_n_minus_1;
  q = L_n_plus_1 - L_n_minus_2;
  q_prime = L_prime_n_plus_1 - L_prime_n_minus_2;
  L = L_n;
}

}  // namespace

/// \cond
template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::Legendre, Quadrature::GaussLobatto>(
    const size_t num_points) noexcept {
  // Algorithm 25 from Kopriva's book, p. 66
  ASSERT(num_points >= 2,
         "Legendre-Gauss-Lobatto quadrature requires at least two collocation "
         "points.");
  const size_t poly_degree = num_points - 1;
  DataVector x(num_points);
  DataVector w(num_points);
  switch (poly_degree) {
    case 1:
      x[0] = -1.;
      x[1] = 1.;
      w[0] = w[1] = 1.;
      break;
    default:
      x[0] = -1.;
      x[poly_degree] = 1.;
      w[0] = w[poly_degree] = 2. / (poly_degree * (poly_degree + 1));
      auto newton_raphson_step = [poly_degree](double logical_coord) noexcept {
        const EvaluateQandL q_and_L(poly_degree, logical_coord);
        return std::make_pair(q_and_L.q, q_and_L.q_prime);
      };
      for (size_t j = 1; j < (poly_degree + 1) / 2; j++) {
        double logical_coord = RootFinder::newton_raphson(
            newton_raphson_step,
            // Initial guess
            -cos((j + 0.25) * M_PI / poly_degree -
                 0.375 / (poly_degree * M_PI * (j + 0.25))),
            // Lower and upper bound, and number of desired base-10 digits
            -1., 1., 14);
        const EvaluateQandL q_and_L(poly_degree, logical_coord);
        x[j] = logical_coord;
        x[poly_degree - j] = -logical_coord;
        w[j] = w[poly_degree - j] =
            2. / (poly_degree * (poly_degree + 1) * square(q_and_L.L));
      }
      if (poly_degree % 2 == 0) {
        const EvaluateQandL q_and_L(poly_degree, 0.);
        x[poly_degree / 2] = 0.;
        w[poly_degree / 2] =
            2. / (poly_degree * (poly_degree + 1) * square(q_and_L.L));
      }
      break;
  }
  return std::make_pair(std::move(x), std::move(w));
}
/// \endcond

}  // namespace Spectral
