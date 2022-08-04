// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace intrp {

struct LinearRegressionResult {
  double intercept;
  double slope;
  double delta_intercept;
  double delta_slope;
};

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A linear regression function.
 *
 * A wrapper for gsl linear regression.
 * Fits the data to \f$y = m x + b\f$. Returns a struct
 * containing \f$(b, m, \delta b, \delta m)\f$, where
 * \f$\delta b\f$ and \f$\delta m\f$ are the error bars in \f$b\f$ and
 * \f$m\f$.  The error bars are computed assuming unknown errors in \f$y\f$.
 *
 * linear_regression could be implemented by calling
 * LinearLeastSquares using Order=1, but we choose instead to
 * implement linear_regression in a simpler way, by calling a simpler gsl
 * function that involves no memory allocations, no copying, and no
 * `pow` functions.
 */
template <typename T>
LinearRegressionResult linear_regression(const T& x_values, const T& y_values);
}  // namespace intrp
