// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

#include <gsl/gsl_poly.h>
#include <limits>

#include "Utilities/ErrorHandling/Assert.hpp"

double positive_root(const double a, const double b, const double c) noexcept {
  const auto roots = real_roots(a, b, c);
  ASSERT(roots[0] <= 0.0 and roots[1] >= 0.0,
         "There are two positive roots, " << roots[0] << " and " << roots[1]
         << ", with a=" << a << " b=" << b << " c=" << c);
  return roots[1];
}

std::array<double, 2> real_roots(const double a, const double b,
                                 const double c) noexcept {
  double x0 = std::numeric_limits<double>::signaling_NaN();
  double x1 = std::numeric_limits<double>::signaling_NaN();
  // clang-tidy: value stored ... never read (true if in Release Build)
  // NOLINTNEXTLINE
  const int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
  ASSERT(num_real_roots == 2,
         "There are only " << num_real_roots << " real roots with a=" << a
         << " b=" << b << " c=" << c);
  return {{x0, x1}};
}
