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
  ASSERT(alpha > -1.0, "alpha = " << alpha);
  ASSERT(beta > -1.0, "beta = " << beta);
  // P_0 = 1
  T P_k_minus_2 = make_with_value<T>(x, 1.);
  if (k == 0) {
    return P_k_minus_2;
  }
  // P_k_1 = (2+a+b)/2*x + (a-b)/2
  T P_k_minus_1 = 0.5 * (2.0 + alpha + beta) * x + 0.5 * (alpha - beta);
  if (k == 1) {
    return P_k_minus_1;
  }
  T P_k = make_with_value<T>(x, 0.);
  // 2(n+1)(n+a+b+1)(2n+a+b)P_{n+1} - [(2n+a+b+1)(a-b)(a+b) +
  //  (2n+a+b)(2n+a+b+1)(2n+a+b+2)x]P_n + 2(n+a)(n+b)(2n+a+b+2)P_{n-1} = 0
  for (size_t j = 1; j < k; ++j) {
    const auto n = static_cast<double>(j);
    const double c0 =
        2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta);
    const double c1 =
        (2.0 * n + alpha + beta + 1.0) * (alpha - beta) * (alpha + beta);
    const double c2 = (2.0 * n + alpha + beta) *
                      (2.0 * n + alpha + beta + 1.0) *
                      (2.0 * n + alpha + beta + 2.0);
    const double c3 =
        2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0);
    P_k = ((c1 + c2 * x) * P_k_minus_1 - c3 * P_k_minus_2) / c0;
    P_k_minus_2 = P_k_minus_1;
    P_k_minus_1 = P_k;
  }
  return P_k;
}

template double Jacobi::basis_function_value(const double alpha,
                                             const double beta, const size_t k,
                                             const double& x);
}  // namespace Spectral
