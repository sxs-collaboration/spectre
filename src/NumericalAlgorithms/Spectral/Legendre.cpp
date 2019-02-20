// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Spectral {

// Algorithms to compute Legendre basis functions
// These functions specialize the templates declared in `Spectral.hpp`.

namespace {
template <typename T>
T compute_basis_function_value_impl(const size_t k, const T& x) noexcept {
  // Algorithm 20 in Kopriva, p. 60
  switch (k) {
    case 0:
      return make_with_value<T>(x, 1.);
    case 1:
      return x;
    default:
      T L_k_minus_2 = make_with_value<T>(x, 1.);
      T L_k_minus_1 = x;
      T L_k = make_with_value<T>(x, 0.);
      for (size_t j = 2; j <= k; j++) {
        L_k = ((2. * j - 1.) * x * L_k_minus_1 - (j - 1.) * L_k_minus_2) / j;
        L_k_minus_2 = L_k_minus_1;
        L_k_minus_1 = L_k;
      }
      return L_k;
  }
}
}  // namespace

/// \cond
template <>
DataVector compute_basis_function_value<Basis::Legendre>(
    const size_t k, const DataVector& x) noexcept {
  return compute_basis_function_value_impl(k, x);
}

template <>
double compute_basis_function_value<Basis::Legendre>(const size_t k,
                                                     const double& x) noexcept {
  return compute_basis_function_value_impl(k, x);
}

template <>
DataVector compute_inverse_weight_function_values<Basis::Legendre>(
    const DataVector& x) noexcept {
  return DataVector(x.size(), 1.);
}

template <>
double compute_basis_function_normalization_square<Basis::Legendre>(
    const size_t k) noexcept {
  return 2. / (2. * k + 1.);
}
/// \endcond

// Algorithms to compute Legendre-Gauss quadrature

namespace {
struct LegendrePolynomialAndDerivative {
  LegendrePolynomialAndDerivative(size_t poly_degree, double x) noexcept;
  double L;
  double dL;
};

LegendrePolynomialAndDerivative::LegendrePolynomialAndDerivative(
    const size_t poly_degree, const double x) noexcept {
  // Algorithm 22 in Kopriva, p. 63
  // The cases where `poly_degree` is `0` or `1` are omitted because they are
  // never used.
  double L_N_minus_2 = 1.;
  double L_N_minus_1 = x;
  double dL_N_minus_2 = 0.;
  double dL_N_minus_1 = 1.;
  double L_N = std::numeric_limits<double>::signaling_NaN();
  double dL_N = std::numeric_limits<double>::signaling_NaN();
  for (size_t k = 2; k <= poly_degree; k++) {
    L_N = ((2. * k - 1.) * x * L_N_minus_1 - (k - 1.) * L_N_minus_2) / k;
    dL_N = dL_N_minus_2 + (2. * k - 1.) * L_N_minus_1;
    L_N_minus_2 = L_N_minus_1;
    L_N_minus_1 = L_N;
    dL_N_minus_2 = dL_N_minus_1;
    dL_N_minus_1 = dL_N;
  }
  L = L_N;
  dL = dL_N;
}

}  // namespace

/// \cond
template <>
std::pair<DataVector, DataVector>
compute_collocation_points_and_weights<Basis::Legendre, Quadrature::Gauss>(
    const size_t num_points) noexcept {
  // Algorithm 23 in Kopriva, p.64
  ASSERT(num_points >= 1,
         "Legendre-Gauss quadrature requires at least one collocation point.");
  const size_t poly_degree = num_points - 1;
  DataVector x(num_points);
  DataVector w(num_points);
  switch (poly_degree) {
    case 0:
      x[0] = 0.;
      w[0] = 2.;
      break;
    case 1:
      x[0] = -sqrt(1. / 3.);
      x[1] = -x[0];
      w[0] = w[1] = 1.;
      break;
    default:
      auto newton_raphson_step = [poly_degree](double logical_coord) noexcept {
        const LegendrePolynomialAndDerivative L_and_dL(poly_degree + 1,
                                                       logical_coord);
        return std::make_pair(L_and_dL.L, L_and_dL.dL);
      };
      for (size_t j = 0; j <= (poly_degree + 1) / 2 - 1; j++) {
        double logical_coord = RootFinder::newton_raphson(
            newton_raphson_step,
            // Initial guess
            -cos((2. * j + 1.) * M_PI / (2. * poly_degree + 2.)),
            // Lower and upper bound, and number of desired base-10 digits
            -1., 1., 14);
        const LegendrePolynomialAndDerivative L_and_dL(poly_degree + 1,
                                                       logical_coord);
        x[j] = logical_coord;
        x[poly_degree - j] = -logical_coord;
        w[j] = w[poly_degree - j] =
            2. / (1. - square(logical_coord)) / square(L_and_dL.dL);
      }
      if (poly_degree % 2 == 0) {
        const LegendrePolynomialAndDerivative L_and_dL(poly_degree + 1, 0.);
        x[poly_degree / 2] = 0.;
        w[poly_degree / 2] = 2. / square(L_and_dL.dL);
      }
      break;
  }
  return std::make_pair(std::move(x), std::move(w));
}
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
  // Algorithm 24 in Kopriva, p. 65
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
  // Algorithm 25 in Kopriva, p. 66
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

template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points) noexcept;

template <>
Matrix spectral_indefinite_integral_matrix<Basis::Legendre>(
    const size_t num_points) noexcept {
  // Tridiagonal matrix that gives the indefinite integral modulo a constant
  Matrix indef_int(num_points, num_points, 0.0);
  for (size_t i = 1; i < num_points - 1; ++i) {
    indef_int(i, i - 1) = 1.0 / (2.0 * i - 1.0);
    indef_int(i, i + 1) = -1.0 / (2.0 * i + 3.0);
  }
  if (LIKELY(num_points > 1)) {
    indef_int(num_points - 1, num_points - 2) =
        1.0 / (2.0 * (num_points - 1) - 1.0);
  }

  // Matrix that ensures that BC at left of interval is 0.0
  Matrix constant(num_points, num_points, 0.0);
  double fac = 1.0;
  for (size_t i = 1; i < num_points; ++i) {
    constant(i, i) = 1.0;
    constant(0, i) = fac;
    fac = -fac;
  }
  return constant * indef_int;
}
/// \endcond

}  // namespace Spectral
