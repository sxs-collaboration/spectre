// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta3.hpp"

#include <cmath>

namespace TimeSteppers {

size_t RungeKutta3::order() const { return 3; }

size_t RungeKutta3::error_estimate_order() const { return 2; }

double RungeKutta3::stable_step() const {
  // This is the condition for  y' = -k y  to go to zero.
  return 0.5 * (1. + cbrt(4. + sqrt(17.)) - 1. / cbrt(4. + sqrt(17.)));
}

const RungeKutta::ButcherTableau& RungeKutta3::butcher_tableau() const {
  // See Hesthaven (5.32)
  static const ButcherTableau tableau{
      // Substep times
      {{1}, {1, 2}},
      // Substep coefficients
      {{1.0},
       {1.0 / 4.0, 1.0 / 4.0}},
      // Result coefficients
      {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0},
      // Coefficients for the embedded method for generating an error measure.
      // See e.g. Chapter II.4 of Harrier, Norsett, and Wagner 1993
      {1.0 / 2.0, 1.0 / 2.0, 0.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -5.0 / 6.0},
       {0.0, 0.0, 1.0 / 6.0},
       {0.0, 0.0, 2.0 / 3.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::RungeKutta3::my_PUP_ID = 0;  // NOLINT
