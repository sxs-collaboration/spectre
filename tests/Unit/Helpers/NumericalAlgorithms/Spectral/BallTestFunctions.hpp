// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
/// \endcond

namespace BallTestFunctions {

/*!
 * \brief Product of polynomials regular on a ball.
 *
 * \details Computes $x^{k_x} y^{k_y} z^{k_z}$ where $x = r \sin \theta \cos
 * \phi$, $y = r \sin \theta \sin \phi$ and $z = r \cos \theta$.  The function
 * and its first derivatives are exactly representable by spherical harmonics of
 * order $(L, M)$ if $L > k_x + k_y + k_z$ and $M > k_x + k_y$. The $\phi$
 * derivative is the Pfaffian derivative.
 */
class ProductOfPolynomials {
 public:
  ProductOfPolynomials(size_t pow_x, size_t pow_y, size_t pow_z);
  DataVector operator()(const DataVector& r, const DataVector& theta,
                        const DataVector& phi) const;
  double operator()(double r, double theta, double phi) const;
  DataVector df_dr(const DataVector& r, const DataVector& theta,
                   const DataVector& phi) const;
  DataVector df_dth(const DataVector& r, const DataVector& theta,
                    const DataVector& phi) const;
  DataVector df_dph(const DataVector& r, const DataVector& theta,
                    const DataVector& phi) const;
  double definite_integral() const;

 private:
  size_t pow_x_;
  size_t pow_y_;
  size_t pow_z_;
};
}  // namespace BallTestFunctions
