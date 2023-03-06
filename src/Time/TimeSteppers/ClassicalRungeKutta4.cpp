// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"

#include <utility>

namespace TimeSteppers {

size_t ClassicalRungeKutta4::order() const { return 4; }

size_t ClassicalRungeKutta4::error_estimate_order() const { return 3; }

// The growth function for RK4 is (e.g. page 60 of
// http://www.staff.science.uu.nl/~frank011/Classes/numwisk/ch10.pdf
//
//   g = 1 + mu + mu^2 / 2 + mu^3 / 6 + mu^4 / 24,
//
// where mu = lambda * dt. The equation dy/dt = -lambda * y evolves
// stably if |g| < 1. For lambda=-2, chosen so the stable_step() for
// RK1 (i.e. forward Euler) would be 1, RK4 has a stable step
// determined by inserting mu->-2 dt into the above equation. Finding the
// solutions with a numerical root find yields a stable step of about 1.39265.
double ClassicalRungeKutta4::stable_step() const { return 1.3926467817026411; }

const RungeKutta::ButcherTableau& ClassicalRungeKutta4::butcher_tableau()
    const {
  // See (17.1.3) of Numerical Recipes 3rd Edition
  static const ButcherTableau tableau{
      // Substep times
      {1.0 / 2.0, 1.0 / 2.0, 1.0, 3.0 / 4.0},
      // Substep coefficients
      {{1.0 / 2.0},
       {0.0, 1.0 / 2.0},
       {0.0, 0.0, 1.0},
       {5.0 / 32.0, 7.0 / 32.0, 13.0 / 32.0, -1.0 / 32.0}},
      // Result coefficients
      {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},
      // Coefficients for the embedded method for generating an error measure.
      {-1.0 / 2.0, 7.0 / 3.0, 7.0 / 3.0, 13.0 / 6.0, -16.0 / 3.0},
      // Dense output coefficient polynomials.  Numerical Recipes
      // Eq. (17.2.15). This implements cubic interpolation throughout
      // the step.
      {{0.0, 1.0, -3.0 / 2.0, 2.0 / 3.0},
       {0.0, 0.0, 1.0, -2.0 / 3.0},
       {0.0, 0.0, 1.0, -2.0 / 3.0},
       {0.0, 0.0, 1.0 / 2.0, -1.0 / 3.0},
       {},
       {0.0, 0.0, -1.0, 1.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::ClassicalRungeKutta4::my_PUP_ID = 0;  // NOLINT
