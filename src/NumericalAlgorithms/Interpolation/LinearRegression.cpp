// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/LinearRegression.hpp"

#include <cmath>
#include <gsl/gsl_fit.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace intrp {
template <typename T>
LinearRegressionResult linear_regression(const T& x_values, const T& y_values) {
  ASSERT(x_values.size() == y_values.size(),
         "Size mismatch: " << x_values.size() << " vs " << y_values.size());
  LinearRegressionResult result{0.0, 0.0, 0.0, 0.0};
  double cov01 = 0.0;  // Off-diagonal component of covariance matrix.
  double sumsq = 0.0;  // Sum of the squares of residuals of the best fit.
  gsl_fit_linear(x_values.data(), 1, y_values.data(), 1, x_values.size(),
                 &result.intercept, &result.slope, &result.delta_intercept,
                 &cov01, &result.delta_slope, &sumsq);
  // After gsl_fit_linear is called, result.delta_intercept and
  // result.delta_slope hold diagonal components of the covariance matrix.  But
  // the error bars are the sqrt of those diagonal components, so take those
  // sqrts here.
  result.delta_intercept = sqrt(result.delta_intercept);
  result.delta_slope = sqrt(result.delta_slope);
  return result;
}
}  // namespace intrp

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                       \
  template intrp::LinearRegressionResult intrp::linear_regression( \
      const DTYPE(data) & x_values, const DTYPE(data) & y_values);
GENERATE_INSTANTIATIONS(INSTANTIATE, (std::vector<double>, DataVector))
#undef INSTANTIATE
#undef DTYPE
