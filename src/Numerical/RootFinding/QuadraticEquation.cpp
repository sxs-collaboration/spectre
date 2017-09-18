// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Numerical/RootFinding/QuadraticEquation.hpp"

#include <gsl/gsl_poly.h>
#include <limits>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/ExpectsAndEnsures.hpp"

double positive_root(const double a, const double b, const double c) {
  double x0 = std::numeric_limits<double>::signaling_NaN();
  double x1 = std::numeric_limits<double>::signaling_NaN();
  int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
  ASSERT(num_real_roots == 2, "Assumes that there are two real roots");
  Ensures(x0 <= 0.0 and x1 >= 0.0);
  return x1;
}

std::array<double, 2> real_roots(const double a, const double b,
                                 const double c) {
  double x0 = std::numeric_limits<double>::signaling_NaN();
  double x1 = std::numeric_limits<double>::signaling_NaN();
  int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
  ASSERT(num_real_roots == 2, "Assumes that there are two real roots");
  return {{x0, x1}};
}
