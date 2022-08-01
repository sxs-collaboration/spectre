// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"

#include <algorithm>
#include <deque>
#include <vector>

#include "NumericalAlgorithms/Interpolation/LinearRegression.hpp"

namespace intrp {

double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values) {
  ASSERT(*std::max_element(x_values.begin(), x_values.end()) <= 0.0,
         "predicted_zero_crossing_value assumes that the x values are "
         "non-positive. This assumption is not necessary for the fit, but "
         "it is necessary to make sense of the special treatment of the "
         "return type for when the error bars are large.");

  const auto [intercept, slope, delta_intercept, delta_slope] =
      linear_regression(x_values, y_values);

  // See the doxygen comments for details of the implementation.
  // For now, set all the x's to zero if denominators are zero.
  const double x_best_fit = slope == 0.0 ? 0.0 : -intercept / slope;
  const double x0 =
      slope + delta_slope == 0.0
          ? 0.0
          : -(intercept - delta_intercept) / (slope + delta_slope);
  const double x1 =
      slope + delta_slope == 0.0
          ? 0.0
          : -(intercept + delta_intercept) / (slope + delta_slope);
  const double x2 =
      slope - delta_slope == 0.0
          ? 0.0
          : -(intercept - delta_intercept) / (slope - delta_slope);
  const double x3 =
      slope - delta_slope == 0.0
          ? 0.0
          : -(intercept + delta_intercept) / (slope - delta_slope);
  if ((x_best_fit > 0.0 and x1 > 0.0 and x2 > 0.0 and x3 > 0.0 and x0 > 0.0) or
      (x_best_fit < 0.0 and x1 < 0.0 and x2 < 0.0 and x3 < 0.0 and x0 < 0.0)) {
    return x_best_fit;
  }
  return 0.0;  // That is, assign no crossing value at all
}

DataVector predicted_zero_crossing_value(
    const std::deque<double>& x_values,
    const std::deque<DataVector>& y_values) {
  ASSERT(x_values.size() == y_values.size(),
         "The x_values and y_values must be of the same size");
  DataVector result(y_values.front().size());

  const std::vector<double> tmp_x_values(x_values.begin(), x_values.end());
  std::vector<double> tmp_y_values(x_values.size());
  for (size_t i = 0; i < result.size(); i++) {
    for (size_t j = 0; j < tmp_y_values.size(); j++) {
      tmp_y_values[j] = y_values[j][i];
    }
    result[i] = predicted_zero_crossing_value(tmp_x_values, tmp_y_values);
  }

  return result;
}

}  // namespace intrp
