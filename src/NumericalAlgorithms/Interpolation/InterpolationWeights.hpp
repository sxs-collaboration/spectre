// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
class Matrix;
/// \endcond

namespace intrp {
/*!
 * \brief Computes the matrix for polynomial interpolation of a non-periodic
 * function known at the set of points \f$x_{source}\f$ to the set of points
 * \f$x_{target}\f$
 *
 * \details The algorithm is from \cite Fornberg1998.  The returned matrix
 * \f$M\f$ will have \f$n_{target}\f$ rows and \f$n_{source}\f$ columns so that
 * \f$f_{target} = M f_{source}\f$
 *
 * \note The accuracy of the interpolation will depend upon the number and
 * distribution of the source points.  It is strongly suggested that you
 * carefully investigate the accuracy for your use case.
 */
Matrix fornberg_interpolation_matrix(const DataVector& x_target,
                                     const DataVector& x_source);

/*!
 * \brief Computes the matrix for interpolating a periodic function known at the
 * set of \f$n\f$ equally spaced points on the periodic domain \f$[0, 2 \pi]\f$
 * to the set of points \f$x_{target}\f$
 *
 * \details The returned matrix \f$M\f$ will have \f$n_{target}\f$ rows and
 * \f$n_{source}\f$ columns so that \f$f_{target} = M f_{source}\f$
 * Formally, this computes the sum
 * \f[ n w_j = 1 + 2 \sum_{k=1}^{\lfloor (n-1)/2 \rfloor} \cos(k(x - X_j))
 *     \left[ + \cos\left(\frac{n}{2}(x - X_j)\right) \right] \f]
 * for each target point \f$x\f$, where \f$ X_j = \frac{2 \pi j}{n}\f$, and
 * the term in brackets is evaluated only if \f$n\f$ is even.
 *
 */
Matrix fourier_interpolation_matrix(const DataVector& x_target,
                                    size_t n_source_points);
}  // namespace intrp
