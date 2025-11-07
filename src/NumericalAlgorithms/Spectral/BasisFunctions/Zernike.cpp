// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Jacobi.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Spectral {
template <size_t Dim>
template <typename T>
T Zernike<Dim>::basis_function_value(const size_t n, const size_t m,
                                     const T& r) {
  static_assert(Dim == 2 or Dim == 3);
  ASSERT(n >= m, "m " << m << " must be at most n " << n);
  ASSERT((n + m) % 2 == 0, "m " << m << " plus n " << n << " must be even");
  const size_t k = (n - m) / 2;
  const auto mm = static_cast<double>(m);
  const double beta = Dim == 2 ? mm : mm + 0.5;
  const T x = 2.0 * square(r) - 1.0;
  T result = pow(r, mm);
  result *= Jacobi::basis_function_value(0.0, beta, k, x);
  return result;
}

template double Zernike<2>::basis_function_value(const size_t n, const size_t m,
                                                 const double& x);

template double Zernike<3>::basis_function_value(const size_t n, const size_t m,
                                                 const double& x);
}  // namespace Spectral
