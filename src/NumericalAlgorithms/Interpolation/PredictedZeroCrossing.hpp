// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <vector>

#include "DataStructures/DataVector.hpp"

namespace intrp {

/*!
 * \brief Predicts the zero crossing of a function.
 *
 * Fits a linear function to a set of y_values at different x_values
 * and uses the fit to predict what x_value the y_value zero will be crossed.
 *
 * predicted_zero_crossing treats x=0 in a special way: All of the
 * x_values must be non-positive; one of the x_values is typically (but
 * is not required to be) zero.  In typical usage, x is time, and x=0
 * is the current time, and we are interested in whether the function
 * crosses zero in the past or in the future. If it cannot be
 * determined (within the error bars of the fit) whether the zero
 * crossing occurs for x < 0 versus x > 0, then we return zero.
 * Otherwise we return the best-fit x for when the function crosses
 * zero.
 *
 * \details We fit to a straight line: y = intercept + slope*x.
 * So our best guess is that the function will cross zero at
 * x_best_fit = -intercept/slope.
 *
 * However, the data are assumed to be noisy.  The fit gives us error
 * bars for the slope and the intercept.  Given the error bars, we can
 * compute four limiting crossing values x0, x1, x2, and x3 by using
 * the maximum and minimum possible values of slope and intercept.
 * For example, if we assume slope<0 and intercept>0, then the
 * earliest possible crossing consistent with the error bars is
 * x3=(-intercept+delta_intercept)/(slope-delta_slope) and the latest
 * possible crossing consistent with the error bars is
 * x0=(-intercept-delta_intercept)/(slope+delta_slope).
 *
 * We compute all four crossing values and demand that all of them
 * are either at x>0 (i.e. in the future if x is time) or at x<0
 * (i.e. in the past if x is time).  Otherwise we conclude that we
 * cannot determine even the sign of the crossing value, so we return
 * zero.
 */
double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values);

/*!
 * \brief Predicts the zero crossing of multiple functions.
 *
 * For the ith element of the DataVector inside y_values, calls
 * predicted_zero_crossing_value(x_values,y_values[:][i]), where we
 * have used python-like notation.
 */
DataVector predicted_zero_crossing_value(
    const std::deque<double>& x_values, const std::deque<DataVector>& y_values);

}  // namespace intrp
