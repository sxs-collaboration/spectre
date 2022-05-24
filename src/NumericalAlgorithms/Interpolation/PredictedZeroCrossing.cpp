// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"

#include <vector>

#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"

namespace intrp {

double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values) {
  intrp::LinearLeastSquares<1> predictor{x_values.size()};
  const auto coefficients = predictor.fit_coefficients(x_values, y_values);
  return -coefficients[0]/coefficients[1];
}

}  // namespace intrp
