// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/Chebyshev.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/InverseWeightFunctionValues.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Spectral {

template <typename T>
T Chebyshev::basis_function_value(const size_t k, const T& x) {
  // Algorithm 21 in Kopriva, p. 60
  switch (k) {
    case 0:
      return make_with_value<T>(x, 1.);
    case 1:
      return x;
    default:
      // These values can be computed either through recursion
      // (implemented here), or analytically as `cos(k * acos(x))`.
      // Since the trigonometric form is expensive to compute it is useful only
      // for large k. See Kopriva, section 3.1 (p. 59) and Fig. 3.1 (p. 61) for
      // a discussion.
      T T_k_minus_2 = make_with_value<T>(x, 1.);
      T T_k_minus_1 = x;
      T T_k = make_with_value<T>(x, 0.);
      for (size_t j = 2; j <= k; j++) {
        T_k = 2. * x * T_k_minus_1 - T_k_minus_2;
        T_k_minus_2 = T_k_minus_1;
        T_k_minus_1 = T_k;
      }
      return T_k;
  }
}

double Chebyshev::basis_function_normalization_square(const size_t k) {
  if (k == 0) {
    return M_PI;
  } else {
    return M_PI_2;
  }
}

template <>
DataVector Chebyshev::collocation_points<Quadrature::Gauss>(
    const size_t num_points) {
  // Algorithm 26 in Kopriva, p. 67
  ASSERT(num_points >= 1,
         "Chebyshev-Gauss quadrature requires at least one collocation point.");
  DataVector result(num_points);
  for (size_t j = 0; j < num_points; j++) {
    result[j] = -cos(M_PI_2 * (2. * static_cast<double>(j) + 1.) /
                     static_cast<double>(num_points));
  }
  return result;
}

template <>
DataVector Chebyshev::collocation_points<Quadrature::GaussLobatto>(
    const size_t num_points) {
  // Algorithm 27 in Kopriva, p. 68
  ASSERT(num_points >= 2,
         "Chebyshev-Gauss-Lobatto quadrature requires at least two collocation "
         "points.");
  const size_t poly_degree = num_points - 1;
  DataVector result(num_points);
  for (size_t j = 0; j < num_points; j++) {
    result[j] =
        -cos(M_PI * static_cast<double>(j) / static_cast<double>(poly_degree));
  }
  return result;
}

template <>
DataVector Chebyshev::integration_weights<Quadrature::Gauss>(
    const size_t num_points) {
  return DataVector{num_points, M_PI / static_cast<double>(num_points)};
}

template <>
DataVector Chebyshev::integration_weights<Quadrature::GaussLobatto>(
    const size_t num_points) {
  // Algorithm 27 in Kopriva, p. 68
  ASSERT(num_points >= 2,
         "Chebyshev-Gauss-Lobatto quadrature requires at least two collocation "
         "points.");
  const size_t poly_degree = num_points - 1;
  DataVector result(num_points, M_PI / static_cast<double>(poly_degree));
  result[0] *= 0.5;
  result[poly_degree] *= 0.5;
  return result;
}

Matrix Chebyshev::indefinite_integral_matrix(const size_t num_points) {
  // Tridiagonal matrix that gives the indefinite integral modulo a constant
  Matrix indef_int(num_points, num_points, 0.0);
  if (LIKELY(num_points > 1)) {
    indef_int(1, 0) = 1.0;
  }
  if (LIKELY(num_points > 2)) {
    indef_int(1, 2) = -0.5;
    indef_int(num_points - 1, num_points - 2) =
        1.0 / (2.0 * (num_points - 1.0));
  }
  for (size_t i = 2; i < num_points - 1; ++i) {
    indef_int(i, i - 1) = 1.0 / (2.0 * i);
    indef_int(i, i + 1) = -1.0 / (2.0 * i);
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

Matrix Chebyshev::definite_integral_matrix(const size_t num_points) {
  Matrix result(1, num_points, 0.0);
  for (size_t j = 0; j < num_points; j += 2) {
    result(0, j) = -2.0 / (square(static_cast<double>(j)) - 1.0);
  }
  return result;
}

// Instantiations of function templates defined in the Spectral directory

template <>
DataVector compute_basis_function_value<Basis::Chebyshev>(const size_t k,
                                                          const DataVector& x) {
  return Chebyshev::basis_function_value(k, x);
}

template <>
double compute_basis_function_value<Basis::Chebyshev>(const size_t k,
                                                      const double& x) {
  return Chebyshev::basis_function_value(k, x);
}

template <>
DataVector compute_inverse_weight_function_values<Basis::Chebyshev>(
    const DataVector& x) {
  return sqrt(1. - square(x));
}

template <>
double compute_basis_function_normalization_square<Basis::Chebyshev>(
    const size_t k) {
  return Chebyshev::basis_function_normalization_square(k);
}

template <>
std::pair<DataVector, DataVector>
compute_collocation_points_and_weights<Basis::Chebyshev, Quadrature::Gauss>(
    const size_t num_points) {
  return std::make_pair(
      Chebyshev::collocation_points<Quadrature::Gauss>(num_points),
      Chebyshev::integration_weights<Quadrature::Gauss>(num_points));
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::Chebyshev, Quadrature::GaussLobatto>(const size_t num_points) {
  return std::make_pair(
      Chebyshev::collocation_points<Quadrature::GaussLobatto>(num_points),
      Chebyshev::integration_weights<Quadrature::GaussLobatto>(num_points));
}

template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

template <>
Matrix spectral_indefinite_integral_matrix<Basis::Chebyshev>(
    const size_t num_points) {
  return Chebyshev::indefinite_integral_matrix(num_points);
}

template <Basis BasisType>
Matrix spectral_definite_integral_matrix(size_t num_points);

template <>
Matrix spectral_definite_integral_matrix<Basis::Chebyshev>(
    const size_t num_points) {
  return Chebyshev::definite_integral_matrix(num_points);
}
}  // namespace Spectral
