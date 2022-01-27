// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "CylindricalEndcapHelpers.hpp"

#include <cmath>
#include <limits>

#include "Utilities/ConstantExpressions.hpp"

namespace cylindrical_endcap_helpers {

double sin_ax_over_x(const double x, const double a) {
  // sin(ax)/x returns the right thing except if x is zero, so
  // we need to treat only that case as special.
  return x == 0.0 ? a : sin(a * x) / x;
}
DataVector sin_ax_over_x(const DataVector& x, const double a) {
  DataVector result(x);
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = sin_ax_over_x(x[i], a);
  }
  return result;
}
double one_over_x_d_sin_ax_over_x(const double x, const double a) {
  // Evaluates (1/x) d/dx [ sin(ax)/x ], which is the quantity that
  // approaches a finite limit as x approaches zero.
  //
  // Here we need to worry about roundoff. Note that we can expand
  // (1/x) d/dx [ sin(ax)/x ] =
  // a/x^2 [ 1 - 1 - 2(ax)^2/3! + 4(ax)^4/5! - 6(ax)^6/7! + ...],
  // where I kept the "1 - 1" above as a reminder that when evaluating
  // this function directly as (a * cos(ax) - sin(ax) / x) / square(x)
  // there can be significant roundoff because of the "1" in each of the
  // two terms that are subtracted.
  //
  // The relative error in the above expression, if evaluated directly,
  // is 3*eps/(ax)^2 where eps is machine epsilon.  (That expression comes
  // from replacing "1 - 1" with eps and noting that the correct answer
  // is the 2(ax)^2/3! term and eps is an error contribution).
  // This means the error is 100% if (ax)^2 is 3*eps.
  //
  // The solution is to evaluate the series if (ax) is small enough.
  // Suppose we keep up to and including the (ax)^(2n) term in the
  // series.  Then the series is accurate if the (ax)^{2n+2} term (the
  // next term in the series) is small, i.e. if
  // (2n+2)(ax)^{2n+2}/(2n+3)! < eps.
  //
  // For the worst case of (2n+2)(ax)^{2n+2}/(2n+3)! == eps, the direct
  // evaluation still has a relative error of 3*eps/(ax)^2, which evaluates to
  // error = 3*eps* eps^{-1/(n+1)} * ((2n+2)/(2n+3)!)^{1/(n+1)}.
  // This can be rewritten as
  // error = 3 * [eps^n*((2n+2)/(2n+3)!)]^{1/(n+1)}.
  //
  // For certain values of n:
  // n=1    error=3*sqrt(eps/30)               ~ 5e-9
  // n=2    error=3*(eps^2/840)^(1/3)          ~ 7e-12
  // n=3    error=3*(eps^3/45360)^(1/4)        ~ 2e-13
  // n=4    error=3*(eps^4/3991680)^(1/5)      ~ 2e-14
  // n=5    error=3*(eps^5/518918400)^(1/6)    ~ 5e-15
  // n=6    error=3*(eps^6/93405312000)^(1/7)  ~ 1e-15
  //
  // We gain less and less with each order.
  //
  // So here we choose n=3.
  // Then the series above can be rewritten
  // 1/x d/dx [ sin(ax)/x ] = a/x^2 [- 2(ax)^2/3! + 4(ax)^4/5! - 6(ax)^6/7!]
  //                        = -a^3/3 [ 1 - 4*3*(ax)^2/5! + 6*3*(ax)^4/7!]
  const double ax = a * x;
  return pow<8>(ax) < 45360.0 * std::numeric_limits<double>::epsilon()
             ? (-cube(a) / 3.0) *
                   (1.0 + square(ax) * (-0.1 + square(ax) / 280.0))
             : (a * cos(ax) - sin(ax) / x) / square(x);
}
DataVector one_over_x_d_sin_ax_over_x(const DataVector& x, const double a) {
  DataVector result(x);
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = one_over_x_d_sin_ax_over_x(x[i], a);
  }
  return result;
}

}  // namespace cylindrical_endcap_helpers
