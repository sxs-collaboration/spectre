// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Spectral {
/// Clenshaw's algorithm for evaluating linear combinations of
/// basis functions obeying a given three term recurrence relation.
/// See numerical recipes 3rd edition 5.4.2 for details.
/// `alpha` and `beta` define the recursion
/// \f$ f_n(x) = \alpha f_{n-1}(x) + \beta f_{n-2}(x) \f$
/// `f0_of_x` and `f1_of_x` are \f$ f_0(x)\f$ and \f$f_1(x)\f$ respectively
template <typename CoefficientDataType, typename DataType>
DataType evaluate_clenshaw(const std::vector<CoefficientDataType>& coefficients,
                           const DataType& alpha, const DataType& beta,
                           const DataType& f0_of_x, const DataType& f1_of_x) {
  if (coefficients.empty()) {
    return make_with_value<DataType>(f0_of_x, 0.0);
  }

  DataType y_upper{make_with_value<DataType>(f0_of_x, 0.0)};  // y_{N+2} = 0
  DataType new_y_lower;
  // Do the loop to calculate y_1 and y_2 (the accumulate function leaves
  // y_lower_final = y_1, and y_upper = y_2)
  const DataType y_lower_final =
      std::accumulate(coefficients.rbegin(), coefficients.rend() - 1,
                      make_with_value<DataType>(f0_of_x, 0.0),
                      [&y_upper, &alpha, &beta, &new_y_lower](
                          const auto& y_lower, const auto& c_k) {
                        new_y_lower = c_k + alpha * y_lower + beta * y_upper;
                        y_upper = y_lower;
                        return new_y_lower;
                      });
  return f0_of_x * coefficients[0] + f1_of_x * y_lower_final +
         beta * f0_of_x * y_upper;
}

}  // namespace Spectral
