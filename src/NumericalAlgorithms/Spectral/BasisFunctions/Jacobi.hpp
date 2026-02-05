// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
class Matrix;
namespace Spectral {
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {

/*!
 * \ingroup SpectralGroup
 *
 * \brief A collection of helper functions for Jacobi polyomials
 *
 */
class Jacobi {
 public:
  /*!
   * \brief Value of the basis function \f$\Phi^k(x) = P^(\alpha,\beta)_k(x)\f$,
   * implemented from \cite Fornberg1996
   */
  template <typename T>
  static T basis_function_value(double alpha, double beta, size_t k,
                                const T& x);
};
}  // namespace Spectral
