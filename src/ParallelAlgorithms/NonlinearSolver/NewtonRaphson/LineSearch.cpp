// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>
#include <cstddef>

#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace NonlinearSolver::newton_raphson {

double next_step_length(const size_t globalization_iteration_id,
                        const double step_length, const double prev_step_length,
                        const double residual, const double residual_slope,
                        const double next_residual,
                        const double prev_residual) {
  ASSERT(residual_slope < 0.,
         "The residual slope must be negative but is: " << residual_slope);
  ASSERT(step_length > 0.,
         "Step lengths must be positive, but current is " << step_length);
  if (globalization_iteration_id == 0) {
    // First globalization step: minimum of quadratic interpolation
    // Note that the expression 10.3c.1T.1 in DennisSchnabel assumes step_length
    // is 1 for this initial iteration, so it omits the factors of step_length
    return -0.5 * square(step_length) * residual_slope /
           (next_residual - residual - step_length * residual_slope);
  } else {
    ASSERT(prev_step_length > 0.,
           "Step lengths must be positive, but prev is " << prev_step_length);
    // Subsequent globalization steps: minimum of cubic interpolation
    const double f1 = next_residual - residual - step_length * residual_slope;
    const double f2 =
        prev_residual - residual - prev_step_length * residual_slope;
    const double a =
        (f1 / square(step_length) - f2 / square(prev_step_length)) /
        (step_length - prev_step_length);
    const double b = (-f1 * prev_step_length / square(step_length) +
                      f2 * step_length / square(prev_step_length)) /
                     (step_length - prev_step_length);
    if (equal_within_roundoff(a, 0.)) {
      return -0.5 * residual_slope / b;
    } else {
      return (sqrt(square(b) - 3. * a * residual_slope) - b) / (3. * a);
    }
  }
}

}  // namespace NonlinearSolver::newton_raphson
