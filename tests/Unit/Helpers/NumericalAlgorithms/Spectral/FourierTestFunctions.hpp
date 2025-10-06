// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
/// \endcond

namespace FourierTestFunctions {

/*!
 * \brief Product of polynomials regular on the surface of a circle
 *
 * \details Computes \f$ n_x^{k_x} n_y^{k_y} \f$ where \f$n_x = \cos \phi\f$
 * and \f$n_y = \sin \phi\f$.  The function and its first derivatives are
 * exactly representable by Fourier modes of order \f$(M)\f$ if \f$M > k_x +
 * k_y\f$.
 */
class ProductOfPolynomials {
 public:
  ProductOfPolynomials(size_t pow_nx, size_t pow_ny);
  DataVector operator()(const DataVector& phi) const;
  double operator()(double phi) const;
  DataVector df_dph(const DataVector& phi) const;
  double definite_integral() const;
  /*!
   * \brief A modal vector of the Fourier modes
   *
   * \details The modal coefficients are stored in a ModalVector as
   * \f$\{u_0, u_1, u_{-1}, u_2, u_{-2}, \ldots, u_M, u_{-M}\}\f$.
   *
   * The modes can be determined from Equation 18 of \cite Mathar2009
   * \f{align*}{
   * \cos^p \phi \sin^q \phi = \frac{(-1)^{q/2}}{2^{p+q}} \sum_{s=0}^p
   * \sum_{\ell=0}^q \binom{p}{s} \binom{q}{\ell} (-1)^{q-\ell} \times &
   *  \begin{cases}
   *     \cos \left[(2s-p+2\ell-q)\phi\right],\; q \text{ is even} \\
   *     \sin \left[(2s-p+2\ell-q)\phi\right],\; q \text{ is odd}
   *  \end{cases}
   * \f}
   */
  DataVector modes() const;

 private:
  size_t pow_nx_;
  size_t pow_ny_;
};
}  // namespace FourierTestFunctions
