// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <cmath>
#include <cstddef>
// IWYU pragma: no_include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Spectral {

// Algorithms to compute Chebyshev basis functions
// These functions specialize the templates declared in `Spectral.hpp`.

/// \cond
template <>
DataVector compute_basis_function_values<Basis::Chebyshev>(
    const size_t k, const DataVector& x) noexcept {
  // Algorithm 21 in Kopriva, p. 60
  switch (k) {
    case 0:
      return DataVector(x.size(), 1.);
    case 1:
      return x;
    default:
      // These values can be computed either through recursion
      // (implemented here), or analytically as `cos(k * acos(x))`.
      // Since the trigonometric form is expensive to compute it is useful only
      // for large k. See Kopriva, section 3.1 (p. 59) and Fig. 3.1 (p. 61) for
      // a discussion.
      DataVector T_k_minus_2(x.size(), 1.);
      DataVector T_k_minus_1 = x;
      DataVector T_k(x.size());
      for (size_t j = 2; j <= k; j++) {
        T_k = 2 * x * T_k_minus_1 - T_k_minus_2;
        T_k_minus_2 = T_k_minus_1;
        T_k_minus_1 = T_k;
      }
      return T_k;
  }
}

template <>
DataVector compute_inverse_weight_function_values<Basis::Chebyshev>(
    const DataVector& x) noexcept {
  return sqrt(1. - square(x));
}

template <>
double compute_basis_function_normalization_square<Basis::Chebyshev>(
    const size_t k) noexcept {
  if (k == 0) {
    return M_PI;
  } else {
    return M_PI_2;
  }
}
/// \endcond

// Algorithm to compute Chebyshev-Gauss quadrature

/// \cond
template <>
std::pair<DataVector, DataVector>
compute_collocation_points_and_weights<Basis::Chebyshev, Quadrature::Gauss>(
    const size_t num_points) noexcept {
  // Algorithm 26 in Kopriva, p. 67
  ASSERT(num_points >= 1,
         "Chebyshev-Gauss quadrature requires at least one collocation point.");
  const size_t poly_degree = num_points - 1;
  DataVector x(num_points);
  DataVector w(num_points, M_PI / num_points);
  for (size_t j = 0; j < num_points; j++) {
    x[j] = -cos(M_PI_2 * (2 * j + 1) / (poly_degree + 1));
  }
  return std::make_pair(std::move(x), std::move(w));
}
/// \endcond

// Algorithm to compute Chebyshev-Gauss-Lobatto quadrature

/// \cond
template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::Chebyshev, Quadrature::GaussLobatto>(
    const size_t num_points) noexcept {
  // Algorithm 27 in Kopriva, p. 68
  ASSERT(num_points >= 2,
         "Chebyshev-Gauss-Lobatto quadrature requires at least two collocation "
         "points.");
  const size_t poly_degree = num_points - 1;
  DataVector x(num_points);
  DataVector w(num_points, M_PI / poly_degree);
  for (size_t j = 0; j < num_points; j++) {
    x[j] = -cos(M_PI * j / poly_degree);
  }
  w[0] *= 0.5;
  w[num_points - 1] *= 0.5;
  return std::make_pair(std::move(x), std::move(w));
}
/// \endcond

}  // namespace Spectral
