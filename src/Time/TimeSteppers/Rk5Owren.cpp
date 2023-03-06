// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk5Owren.hpp"

namespace TimeSteppers {
Rk5Owren::Rk5Owren(CkMigrateMessage* /*msg*/) {}

size_t Rk5Owren::order() const { return 5; }

size_t Rk5Owren::error_estimate_order() const { return 4; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1). For the fifth order method:
//  alpha_6 = 6 (-5 c3**2 + 2 c3) - 2 c6 beta
//  alpha_7 = 14 c3 c6 beta
// where
//   beta = 20 c3**2 - 15 c3 + 3
// The stability limit as compared to a forward Euler method is given by finding
// the root for |p(-2 z)|-1=0. For forward Euler this is 1.0.
double Rk5Owren::stable_step() const { return 1.5961737362090775; }

const RungeKutta::ButcherTableau& Rk5Owren::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {1.0 / 6.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 2.0, 9.0 / 14.0, 7.0 / 8.0},
      // Substep coefficients
      {{1.0 / 6.0},
       {1.0 / 16.0, 3.0 / 16.0},
       {0.25, -0.75, 1.0},
       {-0.75, 15.0 / 4.0, -3.0, 0.5},
       {369.0 / 1372.0, -243.0 / 343.0, 297.0 / 343.0, 1485.0 / 9604.0,
        297.0 / 4802.0},
       {-133.0 / 4512.0, 1113.0 / 6016.0, 7945.0 / 16544.0, -12845.0 / 24064.0,
        -315.0 / 24064.0, 156065.0 / 198528.0}},
      // Result coefficients
      {83.0 / 945.0, 0.0, 248.0 / 825.0, 41.0 / 180.0, 1.0 / 36.0,
       2401.0 / 38610.0, 6016.0 / 20475.0},
      // Coefficients for the embedded method for generating an error measure.
      {-1.0 / 9.0, 0.0, 40.0 / 33.0, -7.0 / 4.0, -1.0 / 12.0, 343.0 / 198.0,
       0.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -3292.0 / 819.0, 17893.0 / 2457.0, -4969.0 / 819.0,
        596.0 / 315.0},
       {},
       {0.0, 0.0, 5112.0 / 715.0, -43568.0 / 2145.0, 1344.0 / 65.0,
        -1984.0 / 275.0},
       {0.0, 0.0, -123.0 / 52.0, 3161.0 / 234.0, -1465.0 / 78.0, 118.0 / 15.0},
       {0.0, 0.0, -63.0 / 52.0, 1061.0 / 234.0, -413.0 / 78.0, 2.0},
       {0.0, 0.0, -40817.0 / 33462.0, 60025.0 / 50193.0, 2401.0 / 1521.0,
        -9604.0 / 6435.0},
       {0.0, 0.0, 18048.0 / 5915.0, -637696.0 / 53235.0, 96256.0 / 5915.0,
        -48128.0 / 6825.0},
       {0.0, 0.0, -18.0 / 13.0, 75.0 / 13.0, -109.0 / 13.0, 4.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk5Owren::my_PUP_ID = 0;  // NOLINT
