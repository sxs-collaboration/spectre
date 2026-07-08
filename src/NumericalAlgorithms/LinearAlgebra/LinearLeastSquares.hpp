// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace intrp {
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A linear least squares solver
 *
 * A wrapper class for the gsl linear least squares solver which determines
 * the coefficients of best fit for a polynomial of order `Order` given a set
 * of data points `x_values` and `y_values` representing some function y(x).
 * The coefficients can be passed to \ref evaluate_polynomial for
 * interpolation.
 *
 * The form taking and returning vectors performs several fits in sequence,
 * reusing internal allocations for efficiency.
 *
 * The details of the linear least squares solver can be seen here:
 * [GSL documentation](https://www.gnu.org/software/gsl/doc/html/lls.html#).
 */
/// @{
template <size_t Order, typename T>
std::array<double, Order + 1> linear_least_squares(const T& x_values,
                                                   const T& y_values);
template <size_t Order, typename T>
std::vector<std::array<double, Order + 1>> linear_least_squares(
    const T& x_values, const std::vector<T>& y_values);
/// @}
}  // namespace intrp
