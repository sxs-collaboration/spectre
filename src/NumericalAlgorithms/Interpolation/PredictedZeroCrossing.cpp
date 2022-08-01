// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"

#include <deque>
#include <vector>

#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"

namespace intrp {

double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values) {
  intrp::LinearLeastSquares<1> predictor{x_values.size()};
  const auto coefficients = predictor.fit_coefficients(x_values, y_values);
  return -coefficients[0]/coefficients[1];
}

DataVector predicted_zero_crossing_value(
    const std::deque<double>& x_values,
    const std::deque<DataVector>& y_values) {
  ASSERT(x_values.size() == y_values.size(),
         "The x_values and y_values must be of the same size");
  intrp::LinearLeastSquares<1> predictor{x_values.size()};

  DataVector result(y_values.front().size());
  std::deque<double> tmp_y_values(x_values.size());
  for (size_t i = 0; i < result.size(); i++) {
    for (size_t j = 0; j < tmp_y_values.size(); j++) {
      tmp_y_values[j] = y_values[j][i];
    }
    const auto coefficients =
        predictor.fit_coefficients(x_values, tmp_y_values);
    result[i] = -coefficients[0] / coefficients[1];
  }

  return result;
}

}  // namespace intrp
