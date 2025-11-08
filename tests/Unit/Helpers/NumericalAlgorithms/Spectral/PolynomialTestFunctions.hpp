// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace PolynomialTestFunctions {

template <Spectral::Basis basis, Spectral::Quadrature quadrature>
void test_orthogonal_polynomial();

/*!
 * \brief The monomial \f$x^n\f$
 */
class Monomial {
 public:
  explicit Monomial(size_t pow_x);
  DataVector operator()(const DataVector& x) const;
  double operator()(double x) const;
  DataVector df_dx(const DataVector& x) const;
  /// The indefinite integral, with constant of integration chosen so that
  /// the integral is zero at \f$x = -1\f$
  DataVector int_f(const DataVector& x) const;
  /// The definite integral on the interval \f$[-1,1]\f$
  double definite_integral() const;
  /*!
   * \brief A modal vector of modes for the given basis
   *
   * \details The modal coefficients are stored in a modal vector as
   * \f$\{u_0, u_1, \ldots, u_n\}\f$.
   */
  template <Spectral::Basis basis>
  DataVector modes() const;

 private:
  size_t pow_x_;
};
}  // namespace PolynomialTestFunctions
