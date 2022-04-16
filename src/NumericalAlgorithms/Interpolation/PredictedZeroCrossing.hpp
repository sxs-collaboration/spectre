// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

namespace intrp {

/*!
 * \brief Predicts the zero crossing of a function.
 *
 * Fits a linear function to a set of y_values at different x_values
 * and uses the fit to predict what x_value the y_value zero will be crossed.
 */
double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values);

}  // namespace intrp
