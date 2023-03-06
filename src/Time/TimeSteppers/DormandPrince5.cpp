// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/DormandPrince5.hpp"

namespace TimeSteppers {

size_t DormandPrince5::order() const { return 5; }

size_t DormandPrince5::error_estimate_order() const { return 4; }

// The growth function for DP5 is
//
//   g = mu^6 / 600 + \sum_{n=0}^5 mu^n / n!,
//
// where mu = lambda * dt. The equation dy/dt = -lambda * y evolves
// stably if |g| < 1. For lambda=-2, chosen so the stable_step() for
// RK1 (i.e. forward Euler) would be 1, DP5 has a stable step
// determined by inserting mu->-2 dt into the above equation. Finding the
// solutions with a numerical root find yields a stable step of about 1.653.
double DormandPrince5::stable_step() const { return 1.6532839463174733; }

const RungeKutta::ButcherTableau& DormandPrince5::butcher_tableau() const {
  // Coefficients from the Dormand-Prince 5 Butcher tableau
  // (e.g. Sec. 7.2 of \cite NumericalRecipes).
  static const ButcherTableau tableau{
      // Substep times
      {1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0},
      // Substep coefficients
      {{1.0 / 5.0},
       {3.0 / 40.0, 9.0 / 40.0},
       {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0},
       {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0},
       {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
        -5103.0 / 18656.0},
       {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
        11.0 / 84.0}},
      // Result coefficients
      {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
       11.0 / 84.0},
      // Coefficients for the embedded method for generating an error measure.
      {5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
       -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -8048581381.0 / 2820520608.0, 8663915743.0 / 2820520608.0,
        -12715105075.0 / 11282082432.0},
       {},
       {0.0, 0.0, 131558114200.0 / 32700410799.0,
        -68118460800.0 / 10900136933.0, 87487479700.0 / 32700410799.0},
       {0.0, 0.0, -1754552775.0 / 470086768.0, 14199869525.0 / 1410260304.0,
        -10690763975.0 / 1880347072.0},
       {0.0, 0.0, 127303824393.0 / 49829197408.0,
        -318862633887.0 / 49829197408.0, 701980252875.0 / 199316789632.0},
       {0.0, 0.0, -282668133.0 / 205662961.0, 2019193451.0 / 616988883.0,
        -1453857185.0 / 822651844.0},
       {},
       {0.0, 0.0, 40617522.0 / 29380423.0, -110615467.0 / 29380423.0,
        69997945.0 / 29380423.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::DormandPrince5::my_PUP_ID = 0;  // NOLINT
