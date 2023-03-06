// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk4Owren.hpp"

namespace TimeSteppers {
Rk4Owren::Rk4Owren(CkMigrateMessage* /*msg*/) {}

size_t Rk4Owren::order() const { return 4; }

size_t Rk4Owren::error_estimate_order() const { return 3; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1). For the fourth order method:
//  alpha_5 = 5 (1 - 2 c_3) c_4
// The stability limit as compared to a forward Euler method is given by finding
// the root for |p(-2 z)|-1=0. For forward Euler this is 1.0.
double Rk4Owren::stable_step() const { return 1.4367588951002057; }

const RungeKutta::ButcherTableau& Rk4Owren::butcher_tableau() const {
  static const ButcherTableau tableau{
      // Substep times
      {1.0 / 6.0, 11.0 / 37.0, 11.0 / 17.0, 13.0 / 15.0},
      // Substep coefficients
      {{1.0 / 6.0},
       {44.0 / 1369.0, 363.0 / 1369.0},
       {3388.0 / 4913.0, -8349.0 / 4913.0, 8140.0 / 4913.0},
       {-36764.0 / 408375.0, 767.0 / 1125.0, -32708.0 / 136125.0,
        210392.0 / 408375.0}},
      // Result coefficients
      {1697.0 / 18876.0, 0.0, 50653.0 / 116160.0, 299693.0 / 1626240.0,
       3375.0 / 11648.0},
      // Coefficients for the embedded method for generating an error measure.
      {101.0 / 363.0, 0.0, -1369.0 / 14520.0, 11849.0 / 14520.0, 0.0},
      // Dense output coefficient polynomials
      {{0.0, 1.0, -104217.0 / 37466.0, 1806901.0 / 618189.0,
        -866577.0 / 824252.0},
       {},
       {0.0, 0.0, 861101.0 / 230560.0, -2178079.0 / 380424.0,
        12308679.0 / 5072320.0},
       {0.0, 0.0, -63869.0 / 293440.0, 6244423.0 / 5325936.0,
        -7816583.0 / 10144640.0},
       {0.0, 0.0, -1522125.0 / 762944.0, 982125.0 / 190736.0,
        -624375.0 / 217984.0},
       {0.0, 0.0, 165.0 / 131.0, -461.0 / 131.0, 296.0 / 131.0}}};
  return tableau;
}
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk4Owren::my_PUP_ID = 0;  // NOLINT
