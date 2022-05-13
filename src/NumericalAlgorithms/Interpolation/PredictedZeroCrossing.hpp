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
 */
double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values);

/*!
 * \brief Predicts the zero crossing of multiple functions contained in a deque
 * of DataVectors.
 *
 * Fits a set of N linear functions, where N is the size of the returned
 * DataVector and the size of each DataVector in the y_values.
 * Each linear function is fit by M points, where M is the size of x_values and
 * y_values. The Nth point in the returned DataVector is the value of x for
 * which the linear fit to the points (x[:], y[:][N]) [using python notation]
 * crosses y = 0. The x_values contain M points in the independent variable,
 * while the y_values contain M DataVectors that each contain (for example) the
 * set of points on a Strahlkorper. Each element of y_values is a DataVector of
 * size N. The first index to y_values selects a DataVector of points
 * corresponding to an x_value, and the second index selects a point in the
 * DataVector, so the y_values are indexed by
 * [x_value_index][point_in_datavector]. The x_values and y_values must be of
 * size M. Each fit determines the zero-crossing for one of the sets of points
 * in the y_values.
 */
DataVector predicted_zero_crossing_value(
    const std::deque<double>& x_values, const std::deque<DataVector>& y_values);

}  // namespace intrp
