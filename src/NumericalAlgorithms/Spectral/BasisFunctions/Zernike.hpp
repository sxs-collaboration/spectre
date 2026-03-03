// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"

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
 * \brief A collection of helper functions for the radial functions used in
 * Zernike polyomials
 *
 */
template <size_t Dim>
class Zernike {
 public:
  /*!
   * \brief Value of the basis function \f$\Phi^m_n(\xi) = R^m_n(r)\f$,
   * where \f$r \equiv \frac{1}{2} (\xi + 1)\f$, implemented from
   * \cite Matsushima1995
   */
  template <typename T>
  static T basis_function_value(size_t n, size_t m, const T& xi);
};
}  // namespace Spectral
