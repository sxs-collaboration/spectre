// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/Jacobi.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Spectral {
template <typename T>
T Jacobi::basis_function_value(const double alpha, const double beta,
                               const size_t k, const T& x) {
  ASSERT(alpha > -1.0, "alpha must be > -1, got " << alpha);
  ASSERT(beta > -1.0, "beta must be > -1, got " << beta);
  // P_0 = 1
  T P_k_minus_2 = make_with_value<T>(x, 1.);
  if (k == 0) {
    return P_k_minus_2;
  }
  // P_1 = (2+a+b)/2*x + (a-b)/2
  T P_k_minus_1 = 0.5 * (2.0 + alpha + beta) * x + 0.5 * (alpha - beta);
  if (k == 1) {
    return P_k_minus_1;
  }
  T P_k = make_with_value<T>(x, 0.);
  // Recurrence from Apendix A of Fornberg1996, taking n+1 -> n
  // 2n(n+a+b)(2n+a+b-2)P_{n} - [(2n+a+b-1)(a-b)(a+b) +
  //  (2n+a+b-2)(2n+a+b-1)(2n+a+b)x]P_{n-1} + 2(n+a-1)(n+b-1)(2n+a+b)P_{n-2} = 0
  // We solve the recurrence relation by defining the coefficients as
  // c_0 P_n - (c_1 + c_2 x) P_{n-1} - c_3 P_{n-2}
  for (size_t j = 2; j <= k; ++j) {
    const auto n = static_cast<double>(j);
    const double c0 =
        2.0 * n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
    const double c1 =
        (2.0 * n + alpha + beta - 1.0) * (alpha - beta) * (alpha + beta);
    const double c2 = (2.0 * n + alpha + beta - 2.0) *
                      (2.0 * n + alpha + beta - 1.0) * (2.0 * n + alpha + beta);
    const double c3 =
        2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * (2.0 * n + alpha + beta);
    P_k = ((c1 + c2 * x) * P_k_minus_1 - c3 * P_k_minus_2) / c0;
    P_k_minus_2 = P_k_minus_1;
    P_k_minus_1 = P_k;
  }
  return P_k;
}

template double Jacobi::basis_function_value<double>(const double alpha,
                                                     const double beta,
                                                     const size_t k,
                                                     const double& x);
template DataVector Jacobi::basis_function_value<DataVector>(
    const double alpha, const double beta, const size_t k, const DataVector& x);
}  // namespace Spectral
