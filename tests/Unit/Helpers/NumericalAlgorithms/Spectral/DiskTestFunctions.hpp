// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
/// \endcond

namespace DiskTestFunctions {

/*!
 * \brief Product of polynomials regular on the surface of a disk
 *
 * \details Computes \f$ n_x^{k_x} n_y^{k_y} \f$ where \f$n_x = r \cos \phi\f$
 * and \f$n_y = r \sin \phi\f$.  The function and its first derivatives are
 * exactly representable by Fourier modes of order \f$(M)\f$ if \f$M > k_x +
 * k_y\f$.
 */
class ProductOfPolynomials {
 public:
  ProductOfPolynomials(size_t pow_nx, size_t pow_ny);
  DataVector operator()(const DataVector& r, const DataVector& phi) const;
  double operator()(double r, double phi) const;
  DataVector df_dr(const DataVector& r, const DataVector& phi) const;
  DataVector df_dph(const DataVector& r, const DataVector& phi) const;
  double definite_integral() const;

 private:
  size_t pow_nx_;
  size_t pow_ny_;
};
}  // namespace DiskTestFunctions
